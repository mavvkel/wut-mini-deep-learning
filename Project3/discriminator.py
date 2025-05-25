import torch
import torch.nn as nn
from generator import CHANNELS


DISCRIMINATOR_FEATURE_MAPS_SIZE = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input size: 3 x 64 x 64
            nn.Conv2d(
                in_channels=CHANNELS,
                out_channels=DISCRIMINATOR_FEATURE_MAPS_SIZE,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(DISCRIMINATOR_FEATURE_MAPS_SIZE),
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

            # input size: (DISCRIMINATOR_FEATURE_MAPS_SIZE * 8) x 4 x 4
            nn.Conv2d(
                in_channels=DISCRIMINATOR_FEATURE_MAPS_SIZE * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.05, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.counter = 0

    def forward(self, x):
        if self.training and self.sigma != 0:
            x = x + torch.normal(0, std=self.sigma, size=x.shape)
        return x
