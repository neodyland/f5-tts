import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import torchaudio
from datasets import load_dataset as hf_load_dataset

from einops import rearrange

from model.modules import MelSpec
from text import text_to_sequence


class HFDataset(Dataset):
    def __init__(
        self,
        text: str,
        audio: str,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.mel_spectrogram = MelSpec(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        )
        self.text = text
        self.audio = audio

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row[self.audio]["array"]
        sample_rate = row[self.audio]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row[self.audio]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row[self.audio]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            audio_tensor = resampler(audio_tensor)

        audio_tensor = rearrange(audio_tensor, "t -> 1 t")

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = rearrange(mel_spec, "1 d t -> d t")

        text = torch.tensor(text_to_sequence(row[self.text])[0], dtype=torch.long)

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


# Dynamic Batch Sampler


class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        frames_threshold: int,
        max_samples=0,
        random_seed=None,
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler,
            desc=f"Sorting with sampler... if slow, check whether dataset is provided with duration",
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices,
            desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu",
        ):
            if batch_frames + frame_len <= self.frames_threshold and (
                max_samples == 0 or len(batch) < max_samples
            ):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset: str,
    split="train",
    mel_spec_kwargs: dict = {},
    text: str = "text",
    audio: str = "audio",
) -> HFDataset:
    ds = hf_load_dataset(dataset, split=split)
    train_dataset = HFDataset(hf_dataset=ds, **mel_spec_kwargs, text=text, audio=audio)

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
