import os
import torch
from typing import Tuple

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchaudio

#
# Adapted from https://github.com/pytorch/audio/pull/1215/files#diff-7c4812ed050b7ad76d3982a5e36ff87469be8d5cc6b38a6fe3be2cd6793f2aee
#


source_folder = './dataset/train/audio/_background_noise_'
target_base_folder = './dataset/train/audio'

sample_rate = 16000  
clip_duration = 1 
overlap = 0.2  

clip_samples = int(clip_duration * sample_rate)
stride_samples = int((clip_duration - overlap) * sample_rate)


def slice_and_save(filepath, target_folder):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    filename = os.path.splitext(os.path.basename(filepath))[0]
    output_folder = os.path.join(target_base_folder, filename)

    os.makedirs(output_folder, exist_ok=True)

    total_samples = waveform.size(1)
    for start in range(0, total_samples - clip_samples + 1, stride_samples):
        end = start + clip_samples
        clip = waveform[:, start:end]

        out_path = os.path.join(output_folder, f"{filename}_{start}.wav")
        torchaudio.save(out_path, clip, sample_rate)


for wav_file in os.listdir(source_folder):
    if wav_file.endswith('.wav'):
        filepath = os.path.join(source_folder, wav_file)
        slice_and_save(filepath, target_base_folder)



UNKNOWN_CLASS_IX = 21

CLASS_NAMES_TO_IXS = {
    "bed": UNKNOWN_CLASS_IX,
    "bird": UNKNOWN_CLASS_IX,
    "cat": UNKNOWN_CLASS_IX,
    "dog": UNKNOWN_CLASS_IX,
    "down": 0,
    "eight": 1,
    "five": 2,
    "four": 3,
    "go": 4,
    "happy": UNKNOWN_CLASS_IX,
    "house": UNKNOWN_CLASS_IX,
    "left": 5,
    "marvin": UNKNOWN_CLASS_IX,
    "nine": 6,
    "no": 7,
    "off": 8,
    "on": 9,
    "one": 10,
    "right": 11,
    "seven": 12,
    "sheila": UNKNOWN_CLASS_IX,
    "six": 13,
    "stop": 14,
    "three": 15,
    "tree": 16,
    "two": 17,
    "up": 18,
    "wow": UNKNOWN_CLASS_IX,
    "yes": 19,
    "zero": 20,
}

def load_audio_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)

    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, filename


class AudioFolder(Dataset):
    """Create a Dataset from Local Files.
    Args:
        path (str): Path to the directory where the dataset is found or downloaded.
        suffix (str) : Audio file type, defaulted to ".WAV".
        pattern (str) : Find pathnames matching this pattern. Defaulted to "*/*"
        new_sample_rate (int) : Resample audio to new sample rate specified.
        spectrogram_transform (bool): If `True` transform the audio waveform and returns it
        transformed into a spectrogram tensor.
    """

    def __init__(
            self,
            path: str,
            suffix: str = ".wav",
            pattern: str = "*/*",
            new_sample_rate: int | None = None,
            spectrogram_transform: bool = False,
        ):

        self._path = path
        self._spectrogram_transform = spectrogram_transform
        self._new_sample_rate = new_sample_rate

        walker = sorted(str(p) for p in Path(self._path).glob(f'{pattern}{suffix}'))
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str]:
        """Load the n-th sample from the dataset.
        Args:
            n (int): The index of the file to be loaded
        Returns:
            tuple: ``(waveform, sample_rate, label, filename)``
        """
        fileid = self._walker[n]

        waveform, sample_rate, label, filename = load_audio_item(fileid, self._path)

        if self._new_sample_rate is not None:
            waveform = torchaudio.transforms.Resample(sample_rate, self._new_sample_rate)(waveform)
            sample_rate = self._new_sample_rate
        if self._spectrogram_transform is not None:
            waveform = torchaudio.transforms.Spectrogram()(waveform)

        return waveform, sample_rate, CLASS_NAMES_TO_IXS[label], filename

    def __len__(self) -> int:
        return len(self._walker)


def collate_fn(data):
    _, sample_rates, labels, _ = zip(*data)

    max_len = max([x[0].shape[2] for x in data])
    n_ftrs = data[0][0].shape[1]

    features = torch.zeros((len(data), 1, n_ftrs, max_len))

    labels = torch.tensor(labels)

    for i in range(len(data)):
        orig = data[i][0]
        l = orig.shape[2]

        features[i, :, :, :l] = orig

    return features.float(), labels.long(), sample_rates


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    SPEECH_COMMANDS_DIRECTORY = './prep_dataset'

    trainloader = DataLoader(
        AudioFolder(SPEECH_COMMANDS_DIRECTORY + '/train'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valloader = DataLoader(
        AudioFolder(SPEECH_COMMANDS_DIRECTORY + '/valid'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    testloader = DataLoader(
        AudioFolder(SPEECH_COMMANDS_DIRECTORY + '/test'),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return trainloader, valloader, testloader
