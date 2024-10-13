import time
from argparse import ArgumentParser

import torch
import torchaudio
from accelerate import Accelerator
from einops import rearrange
from ema_pytorch import EMA
from vocos import Vocos

from model import CFM, DiT
from model.utils import (
    get_tokenizer,
    save_spectrogram,
)
from text import text_to_sequence


# --------------------- Dataset Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1

tokenizer = "jtalk"

exp_name = "f5tts_jp"


infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
cfg_strength = 2.0
speed = 1.0
use_truth_duration = False
no_ref_audio = False

# -------------------------------------------------#

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Tokenizer
vocab_char_map, vocab_size = get_tokenizer()


def main() -> None:
    parser = ArgumentParser(
        name="Inference",
        description="推論します。",
    )
    parser.add_argument(
        "text",
        type=str,
        help="テキスト",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="モデルのパス",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPUで推論します。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="EMAを使用します。",
    )

    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        accelerator = Accelerator()
        device = f"cuda:{accelerator.process_index}"

    checkpoint = torch.load(
        args.checkpoint_path, map_location=device, weights_only=True
    )

    model = CFM(
        transformer=DiT(
            **dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
            ),
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    if args.use_ema:
        ema_model = EMA(model, include_online_model=False).to(device)
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        ema_model.copy_params_from_ema_to_model()
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    start = time.time()
    texts, _ = text_to_sequence(args.text)
    audio, sr = torchaudio.load("test.wav")
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)
    ref_audio_len = audio.shape[-1] // hop_length
    duration = ref_audio_len + int(
        ref_audio_len
        / len(text_to_sequence(ref_text)[0])
        * len(text_to_sequence(gen_text)[0])
        / speed
    )

    # Inference
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=[texts],
            duration=duration,
            steps=32,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1.0,
            seed=114514,
        )


    generated = generated[:, ref_audio_len:, :]
    generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms

    save_spectrogram(generated_mel_spec[0].cpu().numpy(), f"output.png")
    torchaudio.save(f"output.wav", generated_wave, target_sample_rate)
    print(f"Generated wav: {generated_wave.shape}")


if __name__ == "__main__":
    main()
