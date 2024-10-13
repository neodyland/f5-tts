from argparse import ArgumentParser

from model import CFM, DiT, Trainer
from model.utils import get_tokenizer
from model.dataset import load_dataset
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    parser = ArgumentParser(prog="Train", description="学習します。")
    # -------------------------- Dataset Settings --------------------------- #

    target_sample_rate = 24000
    n_mel_channels = 100
    hop_length = 256
    parser.add_argument("--dataset", type=str, default="sin2piusc/jsut_ver1.1:train")
    parser.add_argument("--dataset_text_column", type=str, default="sentence")
    parser.add_argument("--dataset_audio_column", type=str, default="audio")
    parser.add_argument("--exp_name", type=str, default="f5tts_jp")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    batch_size_type = "sample"  # "frame" or "sample"
    max_samples = 64  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
    max_grad_norm = 1.0

    epochs = 50  # use linear decay, thus epochs control the slope
    num_warmup_updates = 2000  # warmup steps
    save_per_updates = 50000  # save checkpoint per steps
    last_per_steps = 5000  # save last checkpoint per steps

    # ----------------------------------------------------------------------- #

    vocab_char_map, vocab_size = get_tokenizer()

    mel_spec_kwargs = dict(
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
    )

    e2tts = CFM(
        transformer=DiT(
            **dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
            ),
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = Trainer(
        e2tts,
        epochs,
        f"ckpts/{args.exp_name}",
        args.learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        batch_size=args.batch_size,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        last_per_steps=last_per_steps,
        accelerate_kwargs={"mixed_precision": "bf16"},
    )
    s = args.dataset.split(":")
    if len(s) != 2:
        print("Invalid dataset format. Use 'dataset_name:split'.")
        return
    train_dataset = load_dataset(
        s[0],
        s[1],
        mel_spec_kwargs=mel_spec_kwargs,
        text=args.dataset_text_column,
        audio=args.dataset_audio_column,
    )
    trainer.train(
        train_dataset,
        resumable_with_seed=None,
        num_workers=args.num_workers,
    )  # seed for shuffling dataset


if __name__ == "__main__":
    main()
