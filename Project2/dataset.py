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

UNKNOWN_CLASS_IX = 22

CLASS_NAMES_TO_IXS = {
    "bed": UNKNOWN_CLASS_IX,
    "bird": UNKNOWN_CLASS_IX,
    "cat": UNKNOWN_CLASS_IX,
    "dog": UNKNOWN_CLASS_IX,
    "down": 1,
    "eight": 2,
    "five": 3,
    "four": 4,
    "go": 5,
    "happy": UNKNOWN_CLASS_IX,
    "house": UNKNOWN_CLASS_IX,
    "left": 6,
    "marvin": UNKNOWN_CLASS_IX,
    "nine": 7,
    "no": 8,
    "off": 9,
    "on": 10,
    "one": 11,
    "right": 12,
    "seven": 13,
    "sheila": UNKNOWN_CLASS_IX,
    "six": 14,
    "stop": 15,
    "three": 16,
    "tree": 17,
    "two": 18,
    "up": 19,
    "wow": UNKNOWN_CLASS_IX,
    "yes": 20,
    "zero": 21,
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
    """
       data: is a list of tuples with (waveform, sample_rate, label, filename)
    """

    waveforms, sample_rates, labels, _ = zip(*data)
    max_len = max(waveforms, key=lambda x: x.shape[2]).shape[2]
    n_ftrs = data[0][0].size(1)

    features = torch.zeros((len(data), n_ftrs, max_len))
    labels = torch.tensor(labels)

    for i in range(len(data)):
        l, k = data[i][0].size(2), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((1, k, max_len - l))], dim=2)

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
