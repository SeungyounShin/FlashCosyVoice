"""
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TORCHDYNAMO_DISABLE=1 \
python -m flashcosyvoice.cosyvoice2_hf_ta \
--model_path /home/robin/FlashCosyVoice/CosyVoice2-0.5B \
--prompt_wav /home/robin/ch-llasa-tts-training/assets/joseph_1.wav \
--text "안녕하세요 TTS 테스트 중입니다. 저는 남자 목소리입니다." \
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
from flashcosyvoice.token2wav import Token2wav


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
    final_emb = torch.cat([speech_emb, llm_emb, text_emb], dim=0).to(dtype=model.model.embed_tokens.weight.dtype)
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


def build_input_ids_text_only(
    tokenizer: AutoTokenizer,
    text: str,
    speech_vocab_size: int,
) -> list[int]:
    """
    text2만 입력으로 받아서 오디오를 생성할 수 있도록 input_ids를 구성.
    포맷: [<speech_sos>] + text2(offset) + [<speech_eos>]
    이후 스피치 토큰은 전부 모델이 생성하도록 함.
    """
    # text 토큰은 말뭉치 뒤에 배치되므로 offset 필요
    offset = speech_vocab_size + 2
    text_ids = [tid + offset for tid in tokenizer.encode(text)]
    input_ids = [speech_vocab_size] + text_ids + [speech_vocab_size + 1]  # sos ... eos
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="CosyVoice2 HF-only text->speech runner (no prompt speech tokens)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_wav", type=str, required=True, help="audio1: speaker ref for flow & xvec")
    parser.add_argument("--text", type=str, default="This is a test.")
    parser.add_argument("--out", type=str, default="out.json", help="Output JSON path")
    parser.add_argument("--out_wav", type=str, default="out.wav", help="Output WAV path")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size_flow", type=int, default=1)
    parser.add_argument("--fp16_flow", action="store_true")
    args = parser.parse_args()
    
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = Config(model=args.model_path, hf_config=CosyVoice2LLMConfig())
    hf_model, llm_tokenizer, eos_id = build_hf_model(args.model_path, cfg.hf_config)

    # -------------------- LLM: text2 -> speech tokens --------------------
    input_ids = build_input_ids_text_only(
        tokenizer=llm_tokenizer,
        text=args.text,
        speech_vocab_size=cfg.hf_config.speech_vocab_size,
    )

    print(f"Input IDs (text-only): {input_ids}")
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids_tensor)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=cfg.hf_config.torch_dtype):
        gen = hf_model.generate(
            input_ids=input_ids_tensor,
            attention_mask=attn_mask,
            do_sample=True,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

    # strip prefix and optional eos
    out_ids = gen[0].tolist()[len(input_ids):]
    if eos_id in out_ids:
        out_ids = out_ids[: out_ids.index(eos_id)]

    payload = {
        "prompt_speech_tokens": [],  # 이전 구조와 달리 프롬프트 스피치 토큰 없음
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

    # Token2wav로 바로 합성
    t2w = Token2wav(args.model_path, float16=(cfg.hf_config.fp16_flow or args.fp16_flow))
    audio_bytes = t2w(out_ids, args.prompt_wav)
    with open(args.out_wav, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved: {args.out} and {args.out_wav}")


if __name__ == "__main__":
    main()
