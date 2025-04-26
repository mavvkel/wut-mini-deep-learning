import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from early_stop import EarlyStopping

from torch.utils.tensorboard.writer import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score, multiclass_auroc

from ASR import ASRModel
from AST import ASTModel
from dataset import CLASS_NAMES_TO_IXS, get_dataloaders
from resnet18 import ResNet, resnet18
from collections import Counter

NUM_CLASSES = len(set(CLASS_NAMES_TO_IXS.values()))
EARLY_STOPING_PATIENCE = 5
EARLY_STOPING_DELTA = 0.00001
LR_PATIENCE = 3

trainloader, valloader, testloader = get_dataloaders(32)

counts = Counter()
for _, _, labels, _ in trainloader.dataset:
    counts[labels] += 1


class_counts = Counter({
    21: 11068, 10: 1685, 13: 1674, 14: 1672, 7: 1669, 17: 1669, 9: 1661, 
    12: 1650, 6: 1649, 20: 1649, 4: 1646, 1: 1644, 18: 1635, 19: 1635, 
    8: 1631, 0: 1626, 11: 1626, 2: 1624, 3: 1623, 5: 1623, 15: 1621, 
    16: 1206, 22: 550
})


total_samples = sum(class_counts.values())


weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}

# Normalizacja wag względem maksymalnej wagi (klasa 22 ma najmniej przykładów)
max_weight = max(weights.values())
normalized_weights = {class_idx: weight / max_weight for class_idx, weight in weights.items()}

# Przekształcenie wag do tensora (aby mogły być użyte w CrossEntropyLoss)
weight_tensor = torch.tensor([normalized_weights.get(i, 1.0) for i in range(23)])




def run_training_session(
    model: ResNet | ASTModel | ASRModel,
    optimizer,
    batch_size: int,
    run_number: int,
    learning_rate_factor: float,
):
    print(f"Running config:")
    print(f"\tmodel={model.__class__.__name__}")
    print(f"\toptimizer={optimizer.__class__.__name__}")
    print(f"\tbatch_size={batch_size}")
    print(f"\trun_number={run_number}")
    print(f"\tlearning_rate_factor={learning_rate_factor}")

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

    run_prefix=f"MODEL={model.__class__.__name__}_BS={batch_size}_RUN={run_number}"
    writer = SummaryWriter(comment=run_prefix)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=LR_PATIENCE)
    early_stopping = EarlyStopping(patience=EARLY_STOPING_PATIENCE, delta=EARLY_STOPING_DELTA, verbose=True)

    checkpoint_dir = 'checkpoints'
    checkpoint_subdir = 'checkpoints/'+ run_prefix + '/'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if not os.path.exists(checkpoint_subdir):
        os.mkdir(checkpoint_subdir)

    trainloader, valloader, testloader = get_dataloaders(batch_size)

    epochs_ran = 0
    st = datetime.now()

    for epoch in range(MAX_EPOCHS):
        model = model.train()

        epoch_train_loss = 0.
        total = 0
        correct = 0
        for _, (waveforms, labels, _) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", unit="batch")):
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

        if epoch % 10 == 9:
            print('Saving progress')
            torch.save(model.state_dict(), checkpoint_dir + f"epochs_{epoch+1}.pt")

        train_accuracy = correct / total

        # Validation accuracy
        model = model.eval()

        total = 0
        correct = 0
        epoch_valid_loss = 0.

        for _, (waveforms, labels, _) in enumerate(valloader):
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

        print(f"Epoch {epoch}:")
        print(f"\tTrain: acc={train_accuracy:.6f}, loss={av_train_loss:.8f}")
        print(f"\tVal: acc={val_accuracy:.6f}, loss={av_valid_loss:.8f}")

        writer.add_scalar("Loss/train", av_train_loss, epoch + 1)
        writer.add_scalar("Loss/valid", av_valid_loss, epoch + 1)

        writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
        writer.add_scalar("Accuracy/valid", val_accuracy, epoch + 1)

        writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch)

        # Check early stopping condition
        early_stopping.check_early_stop(av_valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")

            if epoch % 10 != 9:
                print('Saving progress')
                torch.save(model.state_dict(), checkpoint_dir + f"epochs_{epoch+1}.pt")
            break

        scheduler.step(av_valid_loss)

    writer.flush()

    print(f"Training for {epochs_ran} epochs took {datetime.now() - st}")

    # Evaluate metrics on test set
    model = model.eval()

    total = 0
    correct = 0

    for _, (waveforms, labels, _) in enumerate(testloader):
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
        f.write(f"ACC={test_accuracy}, OPT={optimizer.__class__.__name__}, BATCH={batch_size}, RUN={run_number}, LR_FACTOR={learning_rate_factor}\n")


if __name__ == '__main__':

    model = resnet18(num_classes=NUM_CLASSES, in_channels=1)
    #model = ASRModel(num_classes=NUM_CLASSES)
    #model = ASTModel(label_dim=NUM_CLASSES, model_size='tiny224')

    optimizer = optim.Adam(params=model.parameters())
    optimizer = optim.RMSprop(params=model.parameters())

    run_training_session(
        model=model,
        optimizer=optimizer,
        batch_size=64,
        run_number=1,
        learning_rate_factor=0.1,
    )
