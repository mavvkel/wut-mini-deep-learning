from datetime import datetime
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from discriminator import Discriminator
from generator import LATENT_Z_VECTOR_SIZE, Generator
from dataset import get_dataloader

from bcolors import bcolors as bc

import os
import random
import torch
import torch.optim as optim
import torchvision.utils as vutils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_generator(device, checkpoint_G):
    generator_model = Generator().to(device)

    if checkpoint_G is None:
        generator_model.apply(weights_init)
    else:
        generator_model.load_state_dict(
            torch.load(checkpoint_G, map_location=lambda storage, _: storage)
        )

    generator_model.train()

    return generator_model


def init_discriminator(device, checkpoint_D):
    discriminator_model = Discriminator().to(device)

    if checkpoint_D is None:
        discriminator_model.apply(weights_init)
    else:
        discriminator_model.load_state_dict(
            torch.load(checkpoint_D, map_location=lambda storage, _: storage)
        )

    discriminator_model.train()

    return discriminator_model


def noisy_labels(y, p_flip, device):
    ps = torch.full_like(y, fill_value=p_flip, device=device)
    to_flip = torch.bernoulli(ps).bool()

    y[to_flip] = 1 - y[to_flip]

    return y


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

    generator = generator.train()


MAX_EPOCHS = 70

def run_training_session(
    batch_size: int,
    adam_beta1: float,
    G_learning_rate: float,
    D_learning_rate: float,
    seed: int,
    with_noisy_labels: bool = False,
    label_flip_starting_prob: float = 0.0,
    label_flip_prob_decrease_factor: float = 1.,
    checkpoint_D: None | str = None,
    checkpoint_G: None | str = None,
):
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    LR_PATIENCE = 10

    print(f"Running config:")
    print(f"\tbatch_size={batch_size}")
    print(f"\tAdam beta1 param={adam_beta1}")
    print(f"\tG_learning_rate={G_learning_rate}")
    print(f"\tD_learning_rate={D_learning_rate}")
    print(f"\tseed={seed}")

    prefix = f"FixedGwLReLu__{batch_size}__{adam_beta1}__G{G_learning_rate}__D{D_learning_rate}__{seed}"
    if with_noisy_labels:
        prefix = prefix + f"__NL({label_flip_starting_prob},{label_flip_prob_decrease_factor})"

    print(f"\tgenerated run prefix: {prefix}")

    device = (
        'mps' if torch.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'xpu' if torch.xpu.is_available()
        else 'cpu'
    )
    print(f"\nDevice used: {device}")

    generator = init_generator(device, checkpoint_G)
    discriminator = init_discriminator(device, checkpoint_D)

    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, LATENT_Z_VECTOR_SIZE, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    REAL_LABEL = 1.
    FAKE_LABEL = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=D_learning_rate, betas=(adam_beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=G_learning_rate, betas=(adam_beta1, 0.999))

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

    current_flip_prob = label_flip_starting_prob
    st = datetime.now()
    batch_count = len(trainloader)

    for epoch in range(MAX_EPOCHS):
        if checkpoint_G is not None:
            epoch = epoch + MAX_EPOCHS

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
            if with_noisy_labels:
                label = noisy_labels(label, current_flip_prob, device)

            # Forward pass real batch through D
            output = discriminator(real).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
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
            errD_fake.backward()

            optimizerD.step()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(REAL_LABEL)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)

            errG = criterion(output, label)
            errG.backward()

            optimizerG.step()

            with torch.no_grad():
                epoch_lossD += errD
                epoch_lossG += errG
                epoch_D_pred_on_fake += output.mean().item()

        if with_noisy_labels:
            current_flip_prob *= label_flip_prob_decrease_factor

        av_epoch_lossD = epoch_lossD / batch_count
        av_epoch_lossG = epoch_lossG / batch_count

        av_epoch_D_pred_on_real = epoch_D_pred_on_real / batch_count
        av_epoch_D_pred_on_fake = epoch_D_pred_on_fake / batch_count

        print(f"D loss: {bc.WARNING}{av_epoch_lossD}{bc.ENDC}\tG loss: {bc.OKBLUE}{av_epoch_lossG}{bc.ENDC}\tReals: {bc.OKBLUE}{av_epoch_D_pred_on_real}{bc.ENDC}\tFakes: {bc.WARNING}{av_epoch_D_pred_on_fake}{bc.ENDC}")

        schedulerD.step(av_epoch_lossD)
        schedulerG.step(av_epoch_lossG)

        writer.add_scalar("Loss/train generator", av_epoch_lossG, epoch + 1)
        writer.add_scalar("Loss/train discriminator", av_epoch_lossD, epoch + 1)

        writer.add_scalar("Discriminator accuracy/reals", av_epoch_D_pred_on_real, epoch + 1)
        writer.add_scalar("Discriminator accuracy/fakes", av_epoch_D_pred_on_fake, epoch + 1)

        writer.add_scalar("Learning rate/generator", schedulerG.get_last_lr()[-1], epoch + 1)
        writer.add_scalar("Learning rate/discriminator", schedulerD.get_last_lr()[-1], epoch + 1)

        if epoch % 5 == 4:
            torch.save(discriminator.state_dict(), os.path.join(run_dir, f"D_epochs_{epoch+1}.pt"))
            torch.save(generator.state_dict(), os.path.join(run_dir, f"G_epochs_{epoch+1}.pt"))

        # Check how the generator is doing by saving G's output on fixed_noise
        save_generator_performance(generator, fixed_noise, run_gen_progress_dir, epoch)

    print(f"Training for {MAX_EPOCHS} epochs took {datetime.now() - st}")

    writer.flush()


if __name__ == '__main__':
    # Runs with G including LeakyReLu & fixed Conv, two different LRs for G & D
    run_training_session(
        batch_size=64,
        adam_beta1=0.5,
        G_learning_rate=0.0002,
        D_learning_rate=0.00005,

        with_noisy_labels=False,
        label_flip_starting_prob=0.0,
        label_flip_prob_decrease_factor=1.0,

        seed=1,
        #checkpoint_D='./checkpoints/run__128__0.85__0.00015__/D_epochs_25.pt',
        #checkpoint_G='./checkpoints/run__128__0.85__0.00015__/G_epochs_25.pt',
    )
