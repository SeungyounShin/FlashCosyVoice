import json
import torch
import s3tokenizer

def wav2tokens(wav_path: str, model_path: str, use_cuda: bool = True):
    """
    Convert a WAV file into discrete speech tokens using s3tokenizer
    (e.g., speech_tokenizer_v2_25hz.onnx).
    Returns: List[int] tokens, int tokens_len
    """
    # 1) tokenizer 로드 (ONNX)
    tok = s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2.onnx")
    if use_cuda and torch.cuda.is_available():
        tok = tok.cuda().eval()
        device = "cuda"
    else:
        tok = tok.eval()
        device = "cpu"

    # 2) 오디오 로드 & 전처리 (16kHz가 기대 샘플레이트)
    #    s3tokenizer.load_audio가 자동 리샘플링(sr=16000) 수행
    audio = s3tokenizer.load_audio(wav_path, sr=16000)   # shape: [T], torch.float32 (CPU 텐서)

    # 3) 로그 멜 스펙트로그램
    mels = s3tokenizer.log_mel_spectrogram(audio)       # [n_mels, T_mel]
    mels, mels_lens = s3tokenizer.padding([mels])       # pad & lengths → [1, n_mels, T_mel], [1]

    # 4) 양자화(토큰화)
    tokens, tokens_lens = tok.quantize(mels.to(device), mels_lens.to(device))
    # 일반적으로 tokens shape: [B(=1), T_tokens] (단일 코드북인 경우)
    tokens = tokens.squeeze(0).to("cpu")
    tokens_len = int(tokens_lens[0].item())

    # 5) Python list[int]로 변환
    token_list = tokens[:tokens_len].tolist()
    return token_list, tokens_len

if __name__ == "__main__":
    MODEL_DIR = "./CosyVoice2-0.5B"  # speech_tokenizer_v2_25hz.onnx가 있는 폴더
    WAV_PATH  = "/home/robin/ch-llasa-tts-training/assets/esther_intro.wav"

    tokens, n = wav2tokens(WAV_PATH, MODEL_DIR, use_cuda=True)
    print(f"#tokens: {n} (≈ {n/25:.2f}s @ 25Hz)")
    print(tokens) # [3757, 1951, 3647, 750, 1725, 377, 2387, 4955, 5348, 2270, 5643, 2199, 2906, 194, 2756, 300, 4752, 3415, 1342, 1722, 41, 752, 1887, 4678, 4390, 2386, 4636, 6238, 5302, 2163, 1842, 2908, 3643, 4671, 2457, 5843, 5446, 3616, 2556, 5509, 6420, 6098, 5644, 2196, 5429, 3242, 5429, 1973, 1460, 926, 152, 1605, 4299, 6211, 2880, 4616, 2564, 6261, 4074, 2807, 2347, 300, 3647, 3890, 1441, 2148, 2101, 2183, 683, 5057, 4680, 4996, 2395, 3881, 1518, 1188, 4373, 596, 3161, 1686, 999, 229, 79, 638, 475, 2112, 165, 1460, 1466, 476, 4579, 4536, 2879]