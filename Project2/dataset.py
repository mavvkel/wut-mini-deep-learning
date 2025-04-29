import os
import torch
from random import sample, shuffle
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
clip_duration = 1.0  
stride_duration = 0.5  

clip_samples = int(clip_duration * sample_rate)
stride_samples = int(stride_duration * sample_rate)

def slice_and_save(filepath, target_folder):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    filename = os.path.splitext(os.path.basename(filepath))[0]
    output_folder = os.path.join(target_base_folder, filename)
    os.makedirs(output_folder, exist_ok=True)

    total_samples = waveform.size(1)
    idx = 0
    for start in range(0, total_samples - clip_samples + 1, stride_samples):
        end = start + clip_samples
        clip = waveform[:, start:end]

        if clip.shape[1] == clip_samples:
            out_path = os.path.join(output_folder, f"{filename}_{idx}.wav")
            torchaudio.save(out_path, clip, sample_rate)
            idx += 1

# for wav_file in os.listdir(source_folder):
#     if wav_file.endswith('.wav'):
#         filepath = os.path.join(source_folder, wav_file)
#         slice_and_save(filepath, target_base_folder)




UNKNOWN_CLASS_IX = 21
SILENCE_CLASS_IX = 22

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
    "doing_the_dishes": SILENCE_CLASS_IX,
    "dude_miaowing": SILENCE_CLASS_IX,
    "exercise_bike": SILENCE_CLASS_IX,
    "pink_noise": SILENCE_CLASS_IX,
    "running_tap": SILENCE_CLASS_IX,
    "white_noise": SILENCE_CLASS_IX
}


SILENCE_CLASS_IX_2 = 0
COMMAND_CLASS_IX_2 = 1
UNKNOWN_CLASS_IX_2 = 2

CLASS_NAMES_TO_IXS_2 = {
    "bed": UNKNOWN_CLASS_IX_2,
    "bird": UNKNOWN_CLASS_IX_2,
    "cat": UNKNOWN_CLASS_IX_2,
    "dog": UNKNOWN_CLASS_IX_2,
    "down": COMMAND_CLASS_IX_2,
    "eight": COMMAND_CLASS_IX_2,
    "five": COMMAND_CLASS_IX_2,
    "four": COMMAND_CLASS_IX_2,
    "go": COMMAND_CLASS_IX_2,
    "happy": UNKNOWN_CLASS_IX_2,
    "house": UNKNOWN_CLASS_IX_2,
    "left": COMMAND_CLASS_IX_2,
    "marvin": UNKNOWN_CLASS_IX_2,
    "nine": COMMAND_CLASS_IX_2,
    "no": COMMAND_CLASS_IX_2,
    "off": COMMAND_CLASS_IX_2,
    "on": COMMAND_CLASS_IX_2,
    "one": COMMAND_CLASS_IX_2,
    "right": COMMAND_CLASS_IX_2,
    "seven": COMMAND_CLASS_IX_2,
    "sheila": UNKNOWN_CLASS_IX_2,
    "six": COMMAND_CLASS_IX_2,
    "stop": COMMAND_CLASS_IX_2,
    "three": COMMAND_CLASS_IX_2,
    "tree": COMMAND_CLASS_IX_2,
    "two": COMMAND_CLASS_IX_2,
    "up": COMMAND_CLASS_IX_2,
    "wow": UNKNOWN_CLASS_IX_2,
    "yes": COMMAND_CLASS_IX_2,
    "zero": COMMAND_CLASS_IX_2,
    "doing_the_dishes": SILENCE_CLASS_IX_2,
    "dude_miaowing": SILENCE_CLASS_IX_2,
    "exercise_bike": SILENCE_CLASS_IX_2,
    "pink_noise": SILENCE_CLASS_IX_2,
    "running_tap": SILENCE_CLASS_IX_2,
    "white_noise": SILENCE_CLASS_IX_2
}

def load_audio_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)

    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, filename

def filter_paths(paths: list[str], path: str) -> list[str]:
    command_paths = []
    unknown_paths = []
    silence_paths = []
    for class_name, label_ix in CLASS_NAMES_TO_IXS_2.items():
        c_path = os.path.join(path, class_name)[2:]
        if label_ix == COMMAND_CLASS_IX_2:
            class_paths = list(filter(lambda p: p.startswith(c_path), paths))
            command_paths.extend(class_paths)
        elif label_ix == UNKNOWN_CLASS_IX_2:
            class_paths = list(filter(lambda p: p.startswith(c_path), paths))
            unknown_paths.extend(class_paths)
        elif label_ix == SILENCE_CLASS_IX_2:
            class_paths = list(filter(lambda p: p.startswith(c_path), paths))
            silence_paths.extend(class_paths)

    command_paths = sample(command_paths, len(silence_paths))
    unknown_paths = sample(unknown_paths, len(silence_paths))

    filtered = silence_paths + command_paths + unknown_paths
    shuffle(filtered)

    return filtered

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
            subset: bool = False,
        ):

        self._path = path
        self._spectrogram_transform = spectrogram_transform
        self._new_sample_rate = new_sample_rate

        walker = sorted(str(p) for p in Path(self._path).glob(f'{pattern}{suffix}'))
        if subset:
            walker = filter_paths(walker, path)
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
        AudioFolder(SPEECH_COMMANDS_DIRECTORY + '/train', subset=False),
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
