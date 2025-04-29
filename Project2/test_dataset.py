import torchaudio
from dataset import get_dataloaders


def test_get_dataloaders():
    batch_size = 32
    train, valid, test = get_dataloaders(batch_size)

    waveforms, labels, sample_rates = next(iter(train))

    assert waveforms.shape[0] == batch_size
    assert labels.shape[0] == batch_size

    assert all([s == 8000 for s in sample_rates])

    waveforms, labels, sample_rates = next(iter(valid))

    assert waveforms.shape[0] == batch_size
    assert labels.shape[0] == batch_size

    assert all([s == 8000 for s in sample_rates])

    waveforms, labels, sample_rates = next(iter(test))

    assert waveforms.shape[0] == batch_size
    assert labels.shape[0] == batch_size

    assert all([s == 8000 for s in sample_rates])


if __name__ == '__main__':
    print('TESTS: ', end=None)
    #test_get_dataloaders()
    print('.', end=None)

    waveform, sr = torchaudio.load('./prep_dataset_no_silence/train/cat/0ac15fe9_nohash_0.wav')
    waveform2, sr2 = torchaudio.load('./prep_dataset/train/cat/0ac15fe9_nohash_0.wav')
    print()
