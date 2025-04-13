import os
import re
import hashlib

import numpy as np
from scipy import signal
from scipy.io import wavfile

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'train', 'valid', or 'test'.
  """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)

    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
        (MAX_NUM_WAVS_PER_CLASS + 1)) *
        (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'valid'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'test'
    else:
        result = 'train'

    return result

CLASS_NAMES = [
    'down',
    'eight',
    'five',
    'four',
    'go',
    'left',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'six',
    'stop',
    'three',
    'two',
    'up',
    'yes',
    'zero',
]

UKNOWN_SUBCLASSES = [
    'bed',
    'bird',
    'cat',
    'dog',
    'happy',
    'house',
    'marvin',
    'sheila',
    'tree',
    'wow',
]


def preprocess_dataset():
    sets = ['train', 'valid', 'test']
    new_sample_rate = 8000
    filename = 'dataset/train/audio/nine/012c8314_nohash_2.wav';


    # 1. Create processed dataset structure
    PREPROCESSED_DATASET_PATH = os.path.join(os.getcwd(), 'prep_dataset')
    if not os.path.exists(PREPROCESSED_DATASET_PATH):
        os.mkdir(PREPROCESSED_DATASET_PATH)

        for s in sets:
            os.mkdir(os.path.join(PREPROCESSED_DATASET_PATH, s))

    train_instances = 0
    valid_instances = 0
    test_instances = 0

    # 2. Redistribute files based on subset
    for class_name in CLASS_NAMES + UKNOWN_SUBCLASSES:
        class_source_path = os.path.join(os.getcwd(), f"dataset/train/audio/{class_name}/")
        fs = os.listdir(class_source_path)

        print(f"Processing `{class_name}`: {len(fs)} files")

        for s in sets:
            class_target_path = os.path.join(PREPROCESSED_DATASET_PATH, s, class_name)

            if not os.path.exists(class_target_path):
                os.mkdir(class_target_path)

        for j, file in enumerate(fs):
            if file[-4:] != '.wav':
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

            sample_rate, samples = wavfile.read(file_path)
            resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))

            print(f"\t[{j}] Writing `{file_path}` to `{where_to}`")
            wavfile.write(where_to, new_sample_rate, resampled.astype(np.int16))

    print(f"--- Totals ---")
    print(f"\tTrain #: {train_instances}")
    print(f"\tValid #: {valid_instances}")
    print(f"\tTest #: {test_instances}")
    print(f"\n\tAll #: {train_instances + valid_instances + test_instances}")


if __name__ == '__main__':
    preprocess_dataset()
