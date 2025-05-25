import torch.nn as nn


LATENT_Z_VECTOR_SIZE = 100
GENERATOR_FEATURE_MAPS_SIZE = 64
CHANNELS = 3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_Z_VECTOR_SIZE, GENERATOR_FEATURE_MAPS_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS_SIZE * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # state size. ``(GENERATOR_FEATURE_MAPS_SIZE*8) x 4 x 4``
            nn.ConvTranspose2d(GENERATOR_FEATURE_MAPS_SIZE * 8, GENERATOR_FEATURE_MAPS_SIZE * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS_SIZE * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(GENERATOR_FEATURE_MAPS_SIZE*4) x 8 x 8``
            nn.ConvTranspose2d( GENERATOR_FEATURE_MAPS_SIZE * 4, GENERATOR_FEATURE_MAPS_SIZE * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS_SIZE * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(GENERATOR_FEATURE_MAPS_SIZE*2) x 16 x 16``
            nn.ConvTranspose2d( GENERATOR_FEATURE_MAPS_SIZE * 2, GENERATOR_FEATURE_MAPS_SIZE, 4, 2, 1, bias=False),
            nn.BatchNorm2d(GENERATOR_FEATURE_MAPS_SIZE),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # state size. ``(GENERATOR_FEATURE_MAPS_SIZE) x 32 x 32``

            nn.ConvTranspose2d(
                in_channels=GENERATOR_FEATURE_MAPS_SIZE,
                out_channels=CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh()

            # output size: 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
