import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn

from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ASR import ASRModel
from AST import ASTModel
from dataset import CLASS_NAMES_TO_IXS, get_dataloaders
from resnet18 import ResNet, resnet18
from collections import Counter

NUM_CLASSES = len(set(CLASS_NAMES_TO_IXS.values()))
LR_PATIENCE = 3

def calc_weights():
    #trainloader, _, _ = get_dataloaders(32)

    #counts = Counter()
    #for _, _, labels, _ in trainloader.dataset:
    #    counts[labels] += 1

    class_counts = Counter({
        1: 34118, 2:11068, 0: 325
    })

    total_samples = 34118 + 11068 + 325

    weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}

    max_weight = max(weights.values())
    normalized_weights = {class_idx: weight / max_weight for class_idx, weight in weights.items()}

    return torch.tensor([normalized_weights.get(i, 1.0) for i in range(3)])

def run_training_session(
    model: ResNet | ASTModel | ASRModel,
    optimizer,
    batch_size: int,
    run_number: int,
    learning_rate_factor: float,
    suffix: str,
):
    print(f"Running config:")
    print(f"\tmodel={model.__class__.__name__}")
    print(f"\toptimizer={optimizer.__class__.__name__}")
    print(f"\tbatch_size={batch_size}")
    print(f"\trun_number={run_number}")
    print(f"\tlearning_rate_factor={learning_rate_factor}")
    print(f"\tsuffix={suffix}")

    MAX_EPOCHS = 20
    seed = run_number

    device = (
        'mps' if torch.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'xpu' if torch.xpu.is_available()
        else 'cpu'
    )

    model = model.to(device)
    print(f"Device used: {next(model.parameters()).device}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_prefix=f"__{model.__class__.__name__}__{optimizer.__class__.__name__}__{suffix}"
    writer = SummaryWriter(comment=run_prefix)

    #weight_tensor = calc_weights()
    criterion = nn.CrossEntropyLoss()#weight=weight_tensor.to(device))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=LR_PATIENCE)

    checkpoint_dir = 'checkpoints'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    trainloader, valloader, testloader = get_dataloaders(batch_size)

    epochs_ran = 0
    st = datetime.now()

    for epoch in range(MAX_EPOCHS):
        model = model.train()

        epoch_train_loss = 0.
        total = 0
        correct = 0
        for i, (waveforms, labels, _) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", unit="batch")):

            if model.__class__.__name__ in [ASRModel.__name__, ASTModel.__name__]:
                waveforms = waveforms.squeeze(1)

            # Forward
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            output = model(waveforms)
            optimizer.zero_grad()
            loss = criterion(output, labels)

            with torch.no_grad():
                epoch_train_loss += loss

                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum()
                total += labels.size(0)

            # Back Propragation
            loss.backward()

            # Update the weights/parameters
            optimizer.step()

        train_accuracy = correct / total

        # Validation accuracy
        model = model.eval()

        total = 0
        correct = 0
        epoch_valid_loss = 0.

        for _, (waveforms, labels, _) in enumerate(valloader):
            if model.__class__.__name__ in [ASRModel.__name__, ASTModel.__name__]:
                waveforms = waveforms.squeeze(1)

            waveforms = waveforms.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(waveforms)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                total += labels.size(0)

                epoch_valid_loss += criterion(outputs, labels)

        val_accuracy = correct / total

        av_train_loss = epoch_train_loss / total
        av_valid_loss = epoch_valid_loss / total

        epochs_ran += 1

        print(f"Epoch {epoch+1}:")
        print(f"\tTrain: acc={train_accuracy:.6f}, loss={av_train_loss:.8f}")
        print(f"\tVal: acc={val_accuracy:.6f}, loss={av_valid_loss:.8f}")

        writer.add_scalar("Loss/train", av_train_loss, epoch + 1)
        writer.add_scalar("Loss/valid", av_valid_loss, epoch + 1)

        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
        writer.add_scalar("Accuracy/valid", val_accuracy, epoch + 1)

        writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch+1)

        scheduler.step(av_valid_loss)

    writer.flush()

    print(f"Training for {epochs_ran} epochs took {datetime.now() - st}")

    print('Saving progress')
    torch.save(model.state_dict(), checkpoint_dir + f"/{run_prefix}.pt")

    # Evaluate metrics on test set
    model = model.eval()

    total = 0
    correct = 0

    for _, (waveforms, labels, _) in enumerate(testloader):
        if model.__class__.__name__ in [ASRModel.__name__, ASTModel.__name__]:
            waveforms = waveforms.squeeze(1)

        waveforms = waveforms.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(waveforms)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

    test_accuracy = float(correct) / total

    print(f"Test evaluation")
    print(f"\tAccuracy={test_accuracy:.6f}")

    with open(f"./{model.__class__.__name__}_test_metrics.txt", "a+") as f:
        f.write(f"ACC={test_accuracy}\t{run_prefix}\n")


if __name__ == '__main__':
    configs = [
        # 1st row
        (32, 0.001, 0),
        (64, 0.001, 0),
        (128, 0.001, 0),

        (32, 0.0005, 0),
        (64, 0.0005, 0),
        (128, 0.0005, 0),

        (32, 0.0001, 0),
        (64, 0.0001, 0),
        (128, 0.0001, 0),

        ## 2nd row
        (32, 0.001, 0.0001),
        (64, 0.001, 0.0001),
        (128, 0.001, 0.0001),

        (32, 0.0005, 0.0001),
        (64, 0.0005, 0.0001),
        (128, 0.0005, 0.0001),

        (32, 0.0001, 0.0001),
        (64, 0.0001, 0.0001),
        (128, 0.0001, 0.0001),

        ## 3rd row
        (32, 0.001, 0.001),
        (64, 0.001, 0.001),
        (128, 0.001, 0.001),

        (32, 0.0005, 0.001),
        (64, 0.0005, 0.001),
        (128, 0.0005, 0.001),

        (32, 0.0001, 0.001),
        (64, 0.0001, 0.001),
        (128, 0.0001, 0.001),

        # Additional

        # Large LR
        #(32, 0.01, 0),

        #(32, 0.0001, 0.0001),
        #(64, 0.0001, 0.001),
        #(128, 0.0001, 0.001),
    ]

    NUM_CLASSES_2 = 3

    for bs, lr, wd in configs:
        #model = resnet18(num_classes=NUM_CLASSES, in_channels=1)
        model = ASTModel(label_dim=NUM_CLASSES_2, model_size='tiny224')
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        run_training_session(
            model=model,
            optimizer=optimizer,
            batch_size=bs,
            run_number=1,
            learning_rate_factor=0.1,
            suffix=f"__3_CLASS__({bs},{lr},{wd})",
        )
