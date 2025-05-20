from datetime import datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from discriminator import Discriminator
from generator import LATENT_Z_VECTOR_SIZE, Generator
from dataset import get_dataloader

import os
import random
import torch
import torch.optim as optim
import torchvision.utils as vutils


# Set seed for reproducibility
seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_generator(device):
    generator_model = Generator().to(device)
    generator_model.apply(weights_init)

    return generator_model


def init_discriminator(device):
    discriminator_model = Discriminator().to(device)
    discriminator_model.apply(weights_init)

    return discriminator_model


def save_generator_performance(generator, fixed_noise, run_gen_progress_dir, epoch):
    generator = generator.eval()

    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()

        target_path = os.path.join(run_gen_progress_dir, f"epochs_{epoch + 1}.png")
        vutils.save_image(
            fake,
            target_path,
            padding=2,
            normalize=True
        )
        print(f"Saved Generator progress images at `{target_path}`")

    generator = generator.train()

def run_training_session(
    batch_size: int,
    adam_beta1: float,
    learning_rate: float,
):
    MAX_EPOCHS = 40
    LR_PATIENCE = 5

    print(f"Running config:")
    print(f"\tbatch_size={batch_size}")
    print(f"\tAdam beta1 param={adam_beta1}")
    print(f"\tlearning_rate={learning_rate}")

    prefix = f"run__{batch_size}__{adam_beta1}__{learning_rate}__"
    print(f"\tgenerated run prefix: {prefix}")

    device = (
        'mps' if torch.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'xpu' if torch.xpu.is_available()
        else 'cpu'
    )
    print(f"\nDevice used: {device}")

    generator = init_generator(device)
    discriminator = init_discriminator(device)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, LATENT_Z_VECTOR_SIZE, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    REAL_LABEL = 1.
    FAKE_LABEL = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(adam_beta1, 0.999))

    writer = SummaryWriter(comment=prefix)

    schedulerG = ReduceLROnPlateau(optimizerG, mode='min', patience=LR_PATIENCE)
    schedulerD = ReduceLROnPlateau(optimizerD, mode='min', patience=LR_PATIENCE)

    checkpoint_dir = 'checkpoints'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    run_dir = os.path.join(checkpoint_dir, prefix)
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    gen_progress_dir = 'generator_progress'
    if not os.path.exists(gen_progress_dir):
        os.mkdir(gen_progress_dir)

    run_gen_progress_dir = os.path.join(gen_progress_dir, prefix)
    if not os.path.exists(run_gen_progress_dir):
        os.mkdir(run_gen_progress_dir)


    trainloader = get_dataloader(batch_size)

    save_generator_performance(generator, fixed_noise, run_gen_progress_dir, -1)

    generator = generator.train()
    discriminator = discriminator.train()

    st = datetime.now()
    batch_count = len(trainloader)

    for epoch in range(MAX_EPOCHS):
        epoch_lossD = 0.
        epoch_lossG = 0.

        epoch_D_pred_on_real = 0.
        epoch_D_pred_on_fake = 0.

        for _, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", unit="batch")):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            ## Train D with all-real batch
            discriminator.zero_grad()

            # Format batch
            real = data[0].to(device)

            b_size = real.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = discriminator(real).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()

            with torch.no_grad():
                epoch_D_pred_on_real += output.mean().item()

            ## Train D with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, LATENT_Z_VECTOR_SIZE, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(REAL_LABEL)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()

            optimizerG.step()

            with torch.no_grad():
                epoch_lossD += errD
                epoch_lossG += errG
                epoch_D_pred_on_fake += output.mean().item()

        av_epoch_lossD = epoch_lossD / batch_count
        av_epoch_lossG = epoch_lossG / batch_count

        av_epoch_D_pred_on_real = epoch_D_pred_on_real / batch_count
        av_epoch_D_pred_on_fake = epoch_D_pred_on_fake / batch_count

        schedulerD.step(av_epoch_lossD)
        schedulerG.step(av_epoch_lossG)

        writer.add_scalar("Loss/train generator", av_epoch_lossG, epoch + 1)
        writer.add_scalar("Loss/train discriminator", av_epoch_lossD, epoch + 1)

        writer.add_scalar("Discriminator accuracy/reals", av_epoch_D_pred_on_real, epoch + 1)
        writer.add_scalar("Discriminator accuracy/fakes", av_epoch_D_pred_on_fake, epoch + 1)

        writer.add_scalar("Learning rate/generator", schedulerG.get_last_lr()[-1], epoch + 1)
        writer.add_scalar("Learning rate/discriminator", schedulerD.get_last_lr()[-1], epoch + 1)

        if epoch % 5 == 4:
            print('Saving checkpoints')
            torch.save(discriminator.state_dict(), os.path.join(run_dir, f"D_epochs_{epoch+1}.pt"))
            torch.save(generator.state_dict(), os.path.join(run_dir + f"G_epochs_{epoch+1}.pt"))

        # Check how the generator is doing by saving G's output on fixed_noise
        save_generator_performance(generator, fixed_noise, run_gen_progress_dir, epoch)

    print(f"Training for {MAX_EPOCHS} epochs took {datetime.now() - st}")

    writer.flush()


if __name__ == '__main__':
    run_training_session(batch_size=64, adam_beta1=0.9, learning_rate=0.0002)

