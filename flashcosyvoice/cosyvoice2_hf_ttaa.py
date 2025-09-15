"""
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TORCHDYNAMO_DISABLE=1 \
python -m flashcosyvoice.cosyvoice2_hf_ttaa \
--model_path /home/robin/FlashCosyVoice/CosyVoice2-0.5B \
--prompt_wav /home/robin/ch-llasa-tts-training/assets/esther.wav \
--text "안녕하세요 채널톡 AI 에이전트 알프입니다. 무엇을 도와드릴까요?" \
--out out.json \
--out_wav out.wav \
--fp16_flow
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
import torchaudio
import onnxruntime
import torchaudio.compliance.kaldi as kaldi

import s3tokenizer
from flashcosyvoice.config import Config, CosyVoice2LLMConfig, SamplingParams


def build_hf_model(model_path: str, hf_cfg: CosyVoice2LLMConfig) -> tuple[Qwen2ForCausalLM, AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_path, "CosyVoice-BlankEN"), use_fast=False, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = Qwen2Config(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=hf_cfg.num_key_value_heads,
        rms_norm_eps=hf_cfg.rms_norm_eps,
        rope_theta=hf_cfg.rope_theta,
        bos_token_id=hf_cfg.bos_token_id,
        eos_token_id=hf_cfg.eos_token_id,
        tie_word_embeddings=hf_cfg.tie_word_embeddings,
        use_cache=hf_cfg.use_cache,
    )
    model = Qwen2ForCausalLM(cfg).to(device="cuda", dtype=hf_cfg.torch_dtype).eval()

    # ---- Minimal loader for CosyVoice llm.pt into HF model ----
    ckpt = torch.load(os.path.join(model_path, "llm.pt"), map_location="cpu", weights_only=True)
    hidden = hf_cfg.hidden_size

    # 1) Build embed_tokens.weight = [speech(6562); sos+taskid(2); text(151936)]
    speech_emb = ckpt["speech_embedding.weight"]
    if speech_emb.shape[0] != hf_cfg.speech_vocab_size:
        speech_emb = speech_emb[: hf_cfg.speech_vocab_size, :]
    llm_emb = ckpt["llm_embedding.weight"]  # [2, hidden]
    text_emb = ckpt["llm.model.model.embed_tokens.weight"]
    final_emb = torch.cat(
        [speech_emb, llm_emb, text_emb], dim=0
    ).to(dtype=model.model.embed_tokens.weight.dtype)
    model.model.embed_tokens.weight.data.copy_(final_emb)

    # 2) Load transformer weights directly where names match
    for k, v in ckpt.items():
        if not k.startswith("llm.model."):
            continue
        wname = k.replace("llm.model.", "")
        if wname == "model.model.embed_tokens.weight":
            continue  # already handled
        try:
            param = model.get_parameter(wname)
        except Exception:
            continue
        if param.data.shape != v.shape:
            continue
        param.data.copy_(v.to(dtype=param.data.dtype))

    # 3) Integrate speech-only lm_head weights into HF lm_head (first 6562 rows)
    if "llm_decoder.weight" in ckpt:
        dec_w = ckpt["llm_decoder.weight"]
        if dec_w.shape[0] != hf_cfg.speech_vocab_size:
            dec_w = dec_w[: hf_cfg.speech_vocab_size, :]
        model.lm_head.weight.data[: hf_cfg.speech_vocab_size, :].copy_(dec_w.to(dtype=model.lm_head.weight.dtype))
    if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None and "llm_decoder.bias" in ckpt:
        dec_b = ckpt["llm_decoder.bias"]
        if dec_b.shape[0] != hf_cfg.speech_vocab_size:
            dec_b = dec_b[: hf_cfg.speech_vocab_size]
        model.lm_head.bias.data[: hf_cfg.speech_vocab_size].copy_(dec_b.to(dtype=model.lm_head.bias.dtype))

    return model, tokenizer, hf_cfg.eos_token_id


def build_input_ids(
    tokenizer: AutoTokenizer,
    speech_tokenizer: torch.nn.Module,
    prompt_wav: str,
    prompt_text: str,
    text: str,
    speech_vocab_size: int,
) -> tuple[list[int], list[int]]:
    # 1) speech tokens from prompt wav (16k mel -> quantize)
    audio_16k = s3tokenizer.load_audio(prompt_wav, sr=16000)
    log_mel = s3tokenizer.log_mel_spectrogram(audio_16k)
    log_mel_batched, log_mel_lens = s3tokenizer.padding([log_mel])
    with torch.inference_mode():
        speech_tokens, speech_tokens_lens = speech_tokenizer.cuda().eval().quantize(
            log_mel_batched.cuda(), log_mel_lens.cuda()
        )
    speech_ids = speech_tokens[0, : speech_tokens_lens[0].item()].tolist()

    # 2) text tokens with offset
    offset = speech_vocab_size + 2
    prompt_text_ids = [tid + offset for tid in tokenizer.encode(prompt_text)]
    text_ids = [tid + offset for tid in tokenizer.encode(text)]

    # 3) compose input_ids: [<speech_sos>] + prompt_text + text + [<speech_eos>] + speech_ids
    input_ids = [speech_vocab_size] + prompt_text_ids + text_ids + [speech_vocab_size + 1] + speech_ids
    return input_ids, speech_ids


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 HF-only minimal runner (input_ids only)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, default="안녕하세요 저는 채널톡 상담사 알프입니다 무엇을 도와드릴까요?")
    parser.add_argument("--text", type=str, default="This is a test.")
    parser.add_argument("--out", type=str, default="out.json", help="Output JSON path")
    parser.add_argument("--out_wav", type=str, default="out.wav", help="Output WAV path")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size_flow", type=int, default=1)
    parser.add_argument("--fp16_flow", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = Config(model=args.model_path, hf_config=CosyVoice2LLMConfig())
    hf_model, llm_tokenizer, eos_id = build_hf_model(args.model_path, cfg.hf_config)

    # Flow & HiFi-GAN
    flow = CausalMaskedDiffWithXvec()
    if cfg.hf_config.fp16_flow or args.fp16_flow:
        flow.half()
    flow.load_state_dict(torch.load(f"{args.model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
    flow = flow.cuda().eval()

    hift = HiFTGenerator()
    hift_sd = torch.load(f"{args.model_path}/hift.pt", map_location="cpu", weights_only=True)
    hift.load_state_dict({k.replace('generator.', ''): v for k, v in hift_sd.items()}, strict=True)
    hift = hift.cuda().eval()

    # Speaker embedding (campplus.onnx)
    audio_16k = s3tokenizer.load_audio(args.prompt_wav, sr=16000)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_sess = onnxruntime.InferenceSession(os.path.join(args.model_path, "campplus.onnx"), sess_options=option, providers=["CPUExecutionProvider"])
    spk_feat = kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk_emb_np = spk_sess.run(None, {spk_sess.get_inputs()[0].name: spk_feat.unsqueeze(0).cpu().numpy()})[0]
    spk_emb_for_flow = torch.tensor(spk_emb_np, device="cuda")

    # Flow prompt mel (24k)
    audio, sr = torchaudio.load(args.prompt_wav, backend="soundfile")
    audio = audio.mean(dim=0, keepdim=True)
    if sr != 24000:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)(audio)
    prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, 80]
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence([prompt_mel], batch_first=True, padding_value=0)
    prompt_mels_lens_for_flow = torch.tensor([prompt_mels_for_flow.shape[1]], dtype=torch.int32)

    # build input_ids
    speech_tok = s3tokenizer.load_model("speech_tokenizer_v2_25hz")
    input_ids, speech_ids = build_input_ids(
        tokenizer=llm_tokenizer,
        speech_tokenizer=speech_tok,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        text=args.text,
        speech_vocab_size=cfg.hf_config.speech_vocab_size,
    )

    # generate with input_ids only
    print(f"Input IDs: {input_ids}")
    #import pdb; pdb.set_trace()
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids_tensor)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=cfg.hf_config.torch_dtype):
        gen = hf_model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

    # strip prefix and optional eos
    out_ids = gen[0].tolist()[len(input_ids) :]
    if eos_id in out_ids:
        out_ids = out_ids[: out_ids.index(eos_id)]

    payload = {
        "prompt_speech_tokens": input_ids[input_ids.index(cfg.hf_config.speech_vocab_size + 1) + 1 :],
        "generated_speech_tokens": out_ids,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Flow -> mel -> HiFi-GAN -> wav
    flow_inputs = torch.nn.utils.rnn.pad_sequence([
        torch.tensor(speech_ids + out_ids)
    ], batch_first=True, padding_value=0)
    flow_inputs_lens = torch.tensor([len(speech_ids) + len(out_ids)], dtype=torch.int32)

    with torch.amp.autocast("cuda", dtype=torch.float16 if (cfg.hf_config.fp16_flow or args.fp16_flow) else torch.float32):
        batch_generated_mels, batch_generated_mels_lens = flow(
            flow_inputs.cuda(), flow_inputs_lens.cuda(),
            prompt_mels_for_flow.cuda(), prompt_mels_lens_for_flow.cuda(), spk_emb_for_flow.cuda(),
            streaming=False, finalize=True
        )

    mel = batch_generated_mels[0, :, :batch_generated_mels_lens[0].item()].unsqueeze(0).to(torch.float32).cuda()
    wav, _ = hift(speech_feat=mel)
    torchaudio.save(args.out_wav, wav.detach().cpu(), 24000)
    print(f"Saved: {args.out} and {args.out_wav}")


if __name__ == "__main__":
    main() 