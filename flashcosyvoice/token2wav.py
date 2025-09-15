import io

import torch
import torchaudio
import s3tokenizer
import onnxruntime

import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.modules.flow import CausalMaskedDiffWithXvec
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml


class Token2wav():

    def __init__(self, model_path, float16=False):
        self.float16 = float16

        self.audio_tokenizer = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2.onnx").cuda().eval()

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(f"{model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

        self.flow = CausalMaskedDiffWithXvec()
        if float16:
            self.flow.half()
        self.flow.load_state_dict(torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow.cuda().eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.cuda().eval()

    def __call__(self, generated_speech_tokens, prompt_wav):
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
        )[0], device='cuda')

        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).cuda()
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device='cuda')

        # Flow inputs = [prompt_speech_tokens + generated_speech_tokens]
        gen_tokens = torch.tensor(generated_speech_tokens, dtype=torch.int64).tolist()
        prompt_ids = prompt_speech_tokens[0, : prompt_speech_tokens_lens[0].item()].tolist()
        flow_ids = torch.tensor([prompt_ids + gen_tokens], dtype=torch.int64, device='cuda')
        flow_lens = torch.tensor([len(prompt_ids) + len(gen_tokens)], dtype=torch.int32, device='cuda')

        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            batch_generated_mels, batch_generated_mels_lens = self.flow(
                flow_ids, flow_lens,
                prompt_mels, prompt_mels_lens, spk_emb,
                streaming=False, finalize=True
            )

        # Use only generated part after prompt length
        mel = batch_generated_mels[0, :, prompt_mels_lens[0].item(): batch_generated_mels_lens[0].item()].unsqueeze(0).to(torch.float32)

        # import pdb; pdb.set_trace()
        wav, _ = self.hift(speech_feat=mel)
        output = io.BytesIO()
        torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')

        return output.getvalue()

if __name__ == '__main__':
    token2wav = Token2wav('./CosyVoice2-0.5B')

    tokens = [3757, 1951, 3647, 750, 1725, 377, 2387, 4955, 5348, 2270, 5643, 2199, 2906, 194, 2756, 300, 4752, 3415, 1342, 1722, 41, 752, 1887, 4678, 4390, 2386, 4636, 6238, 5302, 2163, 1842, 2908, 3643, 4671, 2457, 5843, 5446, 3616, 2556, 5509, 6420, 6098, 5644, 2196, 5429, 3242, 5429, 1973, 1460, 926, 152, 1605, 4299, 6211, 2880, 4616, 2564, 6261, 4074, 2807, 2347, 300, 3647, 3890, 1441, 2148, 2101, 2183, 683, 5057, 4680, 4996, 2395, 3881, 1518, 1188, 4373, 596, 3161, 1686, 999, 229, 79, 638, 475, 2112, 165, 1460, 1466, 476, 4579, 4536, 2879]
    audio = token2wav(tokens, '/home/robin/ch-llasa-tts-training/assets/joseph_1.wav')
    with open('output.wav', 'wb') as f:
        f.write(audio)
