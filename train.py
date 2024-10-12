from model import CFM, DiT, Trainer
from model.utils import get_tokenizer
from model.dataset import load_dataset
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# -------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256

split = "train"
dataset_name = "sin2piusc/jsut_ver1.1"


# -------------------------- Training Settings -------------------------- #

exp_name = "f5tts_jp"  # F5TTS_Base | E2TTS_Base

learning_rate = 1e-4

batch_size_per_gpu = 4  # 8 GPUs, 8 * 38400 = 307200
batch_size_type = "sample"  # "frame" or "sample"
max_samples = 64  # max sequences per batch if use frame-wise batch_size. we set 32 for small models, 64 for base models
grad_accumulation_steps = 1  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 11  # use linear decay, thus epochs control the slope
num_warmup_updates = 2000  # warmup steps
save_per_updates = 50000  # save checkpoint per steps
last_per_steps = 5000  # save last checkpoint per steps

# ----------------------------------------------------------------------- #


def main():

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
        f"ckpts/{exp_name}",
        learning_rate,
        num_warmup_updates=num_warmup_updates,
        save_per_updates=save_per_updates,
        batch_size=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        last_per_steps=last_per_steps,
        accelerate_kwargs={"mixed_precision": "bf16"},
    )

    train_dataset = load_dataset(
        dataset_name,
        split,
        mel_spec_kwargs=mel_spec_kwargs,
        text="sentence",
        audio="audio",
    )
    trainer.train(
        train_dataset,
        resumable_with_seed=None,
        num_workers=8,
    )  # seed for shuffling dataset


if __name__ == "__main__":
    main()
