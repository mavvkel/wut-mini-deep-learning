import os
import re
import hashlib

import numpy as np
from scipy import signal
from scipy.io import wavfile

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  

def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'valid'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'test'
    else:
        return 'train'

CLASS_NAMES = [
    'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'off', 'on', 
    'one', 'right', 'seven', 'six', 'stop', 'three', 'two', 'up', 'yes', 'zero',
]

UKNOWN_SUBCLASSES = [
    'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
]

SILENCE_CLASSES = [
    "doing_the_dishes", "dude_miaowing", "exercise_bike", "pink_noise", "running_tap", "white_noise",
]

def preprocess_dataset():
    sets = ['train', 'valid', 'test']
    new_sample_rate = 8000

    PREPROCESSED_DATASET_PATH = os.path.join(os.getcwd(), 'prep_dataset')
    os.makedirs(PREPROCESSED_DATASET_PATH, exist_ok=True)

    for s in sets:
        os.makedirs(os.path.join(PREPROCESSED_DATASET_PATH, s), exist_ok=True)

    train_instances = 0
    valid_instances = 0
    test_instances = 0

    all_classes = CLASS_NAMES + UKNOWN_SUBCLASSES + SILENCE_CLASSES

    for class_name in all_classes:
        class_source_path = os.path.join(os.getcwd(), f"dataset/train/audio/{class_name}/")
        fs = os.listdir(class_source_path)

        print(f"Processing `{class_name}`: {len(fs)} files")

        for s in sets:
            class_target_path = os.path.join(PREPROCESSED_DATASET_PATH, s, class_name)
            os.makedirs(class_target_path, exist_ok=True)

        for j, file in enumerate(fs):
            if not file.endswith('.wav'):
                print(f"Unexpected file path found: `{file}`")
                continue

            target_set = which_set(file, 15, 15)

            if target_set == 'train':
                train_instances += 1
            elif target_set == 'valid':
                valid_instances += 1
            else:
                test_instances += 1

            where_to = os.path.join(PREPROCESSED_DATASET_PATH, target_set, class_name, file)
            file_path = os.path.join(class_source_path, file)

            waveform, sr = torchaudio.load(file_path)
            if sr != new_sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, new_sample_rate)

            print(f"\t[{j}] Writing `{file_path}` to `{where_to}`")
            torchaudio.save(where_to, waveform, new_sample_rate)

    print(f"--- Totals ---")
    print(f"\tTrain #: {train_instances}")
    print(f"\tValid #: {valid_instances}")
    print(f"\tTest #: {test_instances}")
    print(f"\n\tAll #: {train_instances + valid_instances + test_instances}")

if __name__ == '__main__':
    preprocess_dataset()
