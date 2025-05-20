import torch.nn as nn
from generator import CHANNELS


DISCRIMINATOR_FEATURE_MAPS_SIZE = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(CHANNELS) x 64 x 64``
            nn.Conv2d(CHANNELS, DISCRIMINATOR_FEATURE_MAPS_SIZE, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(DISCRIMINATOR_FEATURE_MAPS_SIZE) x 32 x 32``
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS_SIZE, DISCRIMINATOR_FEATURE_MAPS_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(DISCRIMINATOR_FEATURE_MAPS_SIZE*2) x 16 x 16``
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 2, DISCRIMINATOR_FEATURE_MAPS_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(DISCRIMINATOR_FEATURE_MAPS_SIZE*4) x 8 x 8``
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 4, DISCRIMINATOR_FEATURE_MAPS_SIZE * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(DISCRIMINATOR_FEATURE_MAPS_SIZE*8) x 4 x 4``
            nn.Conv2d(DISCRIMINATOR_FEATURE_MAPS_SIZE * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
