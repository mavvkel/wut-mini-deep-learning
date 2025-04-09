import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os
import os
import matplotlib.pyplot as plt
from lenet.py import LeNet5
from early_stop.py import EarlyStopping
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

epochs = 100

mean, std = 0.4605, 0.2475

lr, bs, wd, dr, ps, f, m, n = 0.0005, 128, 1e-05, 0.1, 3, 0.2, 0, 'False'

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = datasets.ImageFolder(root="archive/train", transform=transform_train)
test_dataset = datasets.ImageFolder(root="archive/test", transform=transform_test)
valid_dataset = datasets.ImageFolder(root="archive/valid", transform=transform_test)

num_classes = len(train_dataset.classes)



transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = datasets.ImageFolder(root="archive/train", transform=transform_train)
test_dataset = datasets.ImageFolder(root="archive/test", transform=transform_test)
valid_dataset = datasets.ImageFolder(root="archive/valid", transform=transform_test)

num_classes = len(train_dataset.classes)

# [lr, bs, wd, dr, ps, f, m, n]
# hyperparams = [[0.0005, 128, 1e-05, 0.1, 3, 0.2, 0, 'False'], [0.001, 64, 1e-06, 0, 3, 0.2, 0.85, 'True']]

results = {}
for i in range(3):
    label = f"lr={lr}_bs={bs}_wd={wd}_dr={dr}_ps={ps}_f={f}_m={m}"
    print(f"\nTraining with hyper-parameters: {label}, nesterov = {n}\n")
    
    log_dir = f'runs3_final/run_{i}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle = True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    model = LeNet5(dropout_rate=dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=f, patience=ps)

    checkpoint_file = f'checkpoints_final/checkpoint_{label}_last_run_{i}.pth'
    if os.path.exists(checkpoint_file):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)
    else:
        start_epoch = 0  

    best_loss = float("inf")
    counter = 0
    early_stopping = EarlyStopping(patience=10, delta=0.000001, verbose=True)
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
        writer.flush()

        scheduler.step(valid_loss)

        print(f"Epoch: [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, \
            Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            checkpoint_filename  = f'checkpoints_final/checkpoint_{label}_last_run_{i}.pth'
            save_checkpoint(model, optimizer, epoch + 1, valid_loss, checkpoint_filename)

        
        early_stopping.check_early_stop(valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")
            break

    end_time = time.time()
    train_time = end_time - start_time

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    results[label] = {"test_acc": test_acc, 
                        "test_loss": test_loss,
                        "valid_acc": valid_acc,
                        "valid_loss": valid_loss,
                        "train_time": train_time}

    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.2f}%\n")

log_dirs = [f"runs/logs_lenet_best/run_{i}" for i in range(3)]
labels = ['Run 1', 'Run 2', 'Run 3']
colors = ['forestgreen', 'seagreen', 'mediumseagreen']  

def extract_scalars(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(r"Training for $\eta=0.001$, $s=64$, $d=1 \times 10^{-6}$, $\sigma=0$, $p_\eta=3$, $f=0.2$, $m=0.85$", fontsize=14)
for log_dir, label, color in zip(log_dirs, labels, colors):
    steps, acc = extract_scalars(log_dir, 'Accuracy/valid')
    axes[0].plot(steps, acc, label=label, color=color)

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True)

for log_dir, label, color in zip(log_dirs, labels, colors):
    steps, loss = extract_scalars(log_dir, 'Loss/valid')
    axes[1].plot(steps, loss, label=label, color=color)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

transform_flip = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_rotation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_jitter = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_autoaugment = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])



augmentations = {
    "flip": transform_flip,
    "rotation": transform_rotation,
    "jitter": transform_jitter,
    "autoaugment": transform_autoaugment,
}

results_aug = {}

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


for aug_name, transform_train in augmentations.items():
  if aug_name not in results_aug:
    results_aug[aug_name] = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": [],
        "test_acc": None,
        "test_loss": None,
        "train_time": None,
    }
    print(f"\nTraining with augmentations: {aug_name}\n")

    log_dir = f'runs/logs_augs_final/{aug_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    train_dataset = datasets.ImageFolder(root="archive/train", transform=transform_train)
    test_dataset = datasets.ImageFolder(root="archive/test", transform=transform_test)
    valid_dataset = datasets.ImageFolder(root="archive/valid", transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)

    model = LeNet5(dropout_rate=dr).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=f, patience=ps)
    checkpoint_file = f'checkpoints_final/checkpoint_{aug_name}_last_2.pth'
    if os.path.exists(checkpoint_file):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_file)
    else:
        start_epoch = 0 



    best_loss = float("inf")
    early_stopping = EarlyStopping(patience=10, delta=0.000001, verbose=True)
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[-1], epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
        writer.flush()
        results_aug[aug_name]["train_loss"].append(train_loss)
        results_aug[aug_name]["train_acc"].append(train_acc)
        results_aug[aug_name]["valid_loss"].append(valid_loss)
        results_aug[aug_name]["valid_acc"].append(valid_acc)


        scheduler.step(valid_loss)

        print(f"Epoch: [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            checkpoint_filename  = f'checkpoints_final/checkpoint_{aug_name}_last_2.pth'
            save_checkpoint(model, optimizer, epoch + 1, valid_loss, checkpoint_filename)



        early_stopping.check_early_stop(valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch+1}")
            break

    end_time = time.time()
    train_time = end_time - start_time
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
    results_aug[aug_name]["test_loss"] = test_loss
    results_aug[aug_name]["test_acc"] = test_acc
    results_aug[aug_name]["train_time"] = train_time
   

    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Acc: {test_acc:.2f}%\n")


import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = "runs/logs_aug"
val_acc_logs = {}
val_loss_logs = {}

for aug_name in os.listdir(log_dir):
    aug_path = os.path.join(log_dir, aug_name)
    if not os.path.isdir(aug_path):
        continue
    ea = event_accumulator.EventAccumulator(aug_path)
    ea.Reload()

    try:
        val_acc = ea.Scalars('Accuracy/valid')
        val_loss = ea.Scalars('Loss/valid')

        # Usuń postfix '_final' jeśli istnieje
        clean_name = aug_name.replace('_final', '')
        val_acc_logs[clean_name] = [x.value for x in val_acc]
        val_loss_logs[clean_name] = [x.value for x in val_loss]
    except KeyError:
        print(f"No validation data for {aug_name}")

# Tworzenie wykresów
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = plt.cm.tab10.colors

for i, aug_name in enumerate(val_acc_logs):
    epochs = list(range(1, len(val_acc_logs[aug_name]) + 1))
    color = colors[i % len(colors)]
    
    axes[0].plot(epochs, val_acc_logs[aug_name], label=aug_name, color=color)
    axes[1].plot(epochs, val_loss_logs[aug_name], label=aug_name, color=color)

axes[0].set_title('Validation Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Validation Accuracy')
axes[0].legend()

axes[1].set_title('Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Validation Loss')
axes[1].legend()

fig.suptitle('Augmentation vs Model Performance', fontsize=16)
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
plt.rcParams['text.usetex'] = False
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
batch_sizes = [64, 128, 256, 512]
checkpoints_path = "checkpoints"
model_results = {}
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



for bs in batch_sizes:
    model = LeNet5(dropout_rate=0.5).to(device)
    ckpt = torch.load(f"{checkpoints_path}/bs{bs}_bscomp.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in valid_loader:  
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    model_results[bs] = {
        "val_acc": (all_preds == all_labels).mean(),
        "val_loss": nn.CrossEntropyLoss()(torch.tensor(all_probs), torch.tensor(all_labels)).item(),
        "precision": precision_score(all_labels, all_preds, average='macro'),
        "recall": recall_score(all_labels, all_preds, average='macro'),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "auc": roc_auc_score(all_labels, all_probs, multi_class='ovr'),
        "conf_mat": confusion_matrix(all_labels, all_preds),
        "class_acc": (np.diag(confusion_matrix(all_labels, all_preds)) / np.sum(confusion_matrix(all_labels, all_preds), axis=1)),
    }

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for bs in batch_sizes:
    axs[0].bar(str(bs), model_results[bs]["val_acc"], color="seagreen")
    axs[1].bar(str(bs), model_results[bs]["val_loss"], color="mediumseagreen")
axs[0].set_title("Validation Accuracy")
axs[1].set_title("Validation Loss")
fig.suptitle(r"$d=1 \times 10^{-6}$, $m=0.85$, $f=0.2$, $\sigma=0.5$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
for i, bs in enumerate(batch_sizes):
    sns.heatmap(
        model_results[bs]["conf_mat"],
        annot=True,
        fmt='d',
        cmap='pink',
        ax=axs[i],
        xticklabels=class_names,
        yticklabels=class_names
    )
    axs[i].set_title(f'Confusion Matrix (BS={bs})')
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')
fig.suptitle(r"Confusion Matrices: $d=1 \times 10^{-6}$, $m=0.85$, $f=0.2$, $\sigma=0.5$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()
colors = plt.cm.tab10.colors  

for i, bs in enumerate(batch_sizes):
    axs[i].bar(
        range(10),
        model_results[bs]["class_acc"],
        color=colors
    )
    axs[i].set_title(f'Class-wise Accuracy (BS={bs})')
    axs[i].set_ylim(0, 1)
    axs[i].set_xticks(range(10))
    axs[i].set_xticklabels(class_names, rotation=45)

fig.suptitle(r"Class-wise Accuracy: $d=1 \times 10^{-6}$, $m=0.85$, $f=0.2$, $\sigma=0.5$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
metric_names = ["f1", "precision", "recall", "auc"]
plot_titles = ["F1 Score", "Precision", "Recall", "AUC-ROC"]
colors = ['mediumseagreen', 'teal', 'darkcyan', 'cadetblue']

for i, metric in enumerate(metric_names):
    ax = axs[i // 2][i % 2]
    for bs, color in zip(batch_sizes, colors):
        ax.bar(str(bs), model_results[bs][metric], label=f'BS {bs}', color=color)
    ax.set_ylim(0, 1)
    ax.set_title(plot_titles[i])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(metric.capitalize())

fig.suptitle(r"F1 / Precision / Recall / AUC Comparison: $d=1 \times 10^{-6}$, $m=0.85$, $f=0.2$, $\sigma=0.5$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os




dropout_rate = [0, 0.1, 0.5]
checkpoints_path = "checkpoints"
model_results = {}
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


valid_dataset = datasets.ImageFolder(root="archive/valid", transform=transform_test)

valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
for dr in dropout_rate:
    
    model = LeNet5(dropout_rate=dr).to(device)
    ckpt = torch.load(f"{checkpoints_path}/bs64_dr{dr}.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in valid_loader:  
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    model_results[dr] = {
        "val_acc": (all_preds == all_labels).mean(),
        "val_loss": nn.CrossEntropyLoss()(torch.tensor(all_probs), torch.tensor(all_labels)).item(),
        "precision": precision_score(all_labels, all_preds, average='macro'),
        "recall": recall_score(all_labels, all_preds, average='macro'),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "auc": roc_auc_score(all_labels, all_probs, multi_class='ovr'),
        "conf_mat": confusion_matrix(all_labels, all_preds),
        "class_acc": (np.diag(confusion_matrix(all_labels, all_preds)) / np.sum(confusion_matrix(all_labels, all_preds), axis=1)),
    }


fig, axs = plt.subplots(1, 2, figsize=(21, 6))

for dr in dropout_rate:
    axs[0].bar(str(dr), model_results[dr]["val_acc"], color="seagreen")
axs[0].set_title("Validation Accuracy")

for dr in dropout_rate:
    axs[1].bar(str(dr), model_results[dr]["val_loss"], color="mediumseagreen")
axs[1].set_title("Validation Loss")

fig.suptitle(r"$s=64$, $d=1 \times 10^{-4}$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.0005$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(21, 6))
for i, dr in enumerate(dropout_rate[:3]): 
    sns.heatmap(
        model_results[dr]["conf_mat"],
        annot=True,
        fmt='d',
        cmap='pink',
        ax=axs[i],
        xticklabels=class_names,
        yticklabels=class_names
    )
    axs[i].set_title(f'Confusion Matrix ($\\sigma={dr}$)')
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')

fig.suptitle(r"Confusion Matrices: $s=64$, $d=1 \times 10^{-4}$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.0005$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(21, 6))
colors = plt.cm.tab10.colors

for i, dr in enumerate(dropout_rate[:3]):
    axs[i].bar(range(10), model_results[dr]["class_acc"], color=colors)
    axs[i].set_title(f'Class-wise Accuracy ($\\sigma={dr}$)')
    axs[i].set_ylim(0, 1)
    axs[i].set_xticks(range(10))
    axs[i].set_xticklabels(class_names, rotation=45)

fig.suptitle(r"Class-wise Accuracy: $s=64$, $d=1 \times 10^{-4}$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.0005$")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
metric_names = ["f1", "precision", "recall", "auc"]
plot_titles = ["F1 Score", "Precision", "Recall", "AUC-ROC"]
colors = ['mediumseagreen', 'teal', 'darkcyan', 'cadetblue']

for i, metric in enumerate(metric_names):
    ax = axs[i // 2][i % 2]
    for bs, color in zip(dropout_rate, colors):
        ax.bar(str(bs), model_results[bs][metric], label=f'\\sigma={bs}', color=color)
    ax.set_title(plot_titles[i])
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel(metric.capitalize())

fig.suptitle(r"F1 / Precision / Recall / AUC Comparison: $s=64$, $d=1 \times 10^{-4}$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.0005$")
plt.tight_layout()
plt.show()

import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, precision_score,
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
import os



plt.rcParams['text.usetex'] = False
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
weight_decay = ['1e-06', '1e-05', '1e-04']
checkpoints_path = "checkpoints_sgd"
model_results = {}
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
valid_dataset = datasets.ImageFolder(root="archive/valid", transform=transform_test)

valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
for wd in weight_decay:
    
    model = LeNet5(dropout_rate=0).to(device)
    ckpt = torch.load(f"{checkpoints_path}/bs64_wd{wd}.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in valid_loader:  
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    model_results[wd] = {
        "val_acc": (all_preds == all_labels).mean(),
        "val_loss": nn.CrossEntropyLoss()(torch.tensor(all_probs), torch.tensor(all_labels)).item(),
        "precision": precision_score(all_labels, all_preds, average='macro'),
        "recall": recall_score(all_labels, all_preds, average='macro'),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "auc": roc_auc_score(all_labels, all_probs, multi_class='ovr'),
        "conf_mat": confusion_matrix(all_labels, all_preds),
        "class_acc": (np.diag(confusion_matrix(all_labels, all_preds)) / np.sum(confusion_matrix(all_labels, all_preds), axis=1)),
    }


fig, axs = plt.subplots(1, 2, figsize=(21, 6))

# Validation Accuracy
for dr in weight_decay:
    axs[0].bar(str(dr), model_results[dr]["val_acc"], color="seagreen")
axs[0].set_title("Validation Accuracy")

for dr in weight_decay:
    axs[1].bar(str(dr), model_results[dr]["val_loss"], color="mediumseagreen")
axs[1].set_title("Validation Loss")

fig.suptitle(r"$s=64$, $\sigma=0$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(21, 6))
for i, dr in enumerate(weight_decay[:3]):  
    sns.heatmap(
        model_results[dr]["conf_mat"],
        annot=True,
        fmt='d',
        cmap='pink',
        ax=axs[i],
        xticklabels=class_names,
        yticklabels=class_names
    )
    axs[i].set_title(f'Confusion Matrix ($d={dr}$)')
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('True')

fig.suptitle(r"Confusion Matrices: $s=64$, $\sigma=0$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(21, 6))
colors = plt.cm.tab10.colors

for i, dr in enumerate(weight_decay[:3]):
    axs[i].bar(range(10), model_results[dr]["class_acc"], color=colors)
    axs[i].set_title(f'Class-wise Accuracy ($d={dr}$)')
    axs[i].set_ylim(0, 1)
    axs[i].set_xticks(range(10))
    axs[i].set_xticklabels(class_names, rotation=45)

fig.suptitle(r"Class-wise Accuracy: $s=64$, $\sigma=0$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(2, 2, figsize=(14, 10))
metric_names = ["f1", "precision", "recall", "auc"]
plot_titles = ["F1 Score", "Precision", "Recall", "AUC-ROC"]
colors = ['mediumseagreen', 'teal', 'darkcyan', 'cadetblue']

for i, metric in enumerate(metric_names):
    ax = axs[i // 2][i % 2]
    for bs, color in zip(weight_decay, colors):
        ax.bar(str(bs), model_results[bs][metric], label=f'd={bs}', color=color)
    ax.set_title(plot_titles[i])
    ax.set_xlabel("Weight Decay")
    ax.set_ylabel(metric.capitalize())

fig.suptitle(r"F1 / Precision / Recall / AUC Comparison: $s=64$, $\sigma=0$, $m=0.85$, $f=0.2$, $p_\eta=5$, $\eta=0.001$")
plt.tight_layout()
plt.show()

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dirs = [f"runs/logs_lenet_best/run_{i}" for i in range(3)]
val_accuracies_last_epoch = []

def extract_scalars(log_dir, tag):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

for log_dir in log_dirs:
    steps, acc = extract_scalars(log_dir, 'Accuracy/valid')
    val_accuracies_last_epoch.append(acc[-1])  

mean_acc = np.mean(val_accuracies_last_epoch)
std_acc = np.std(val_accuracies_last_epoch)

print(f"Validation Accuracy (last epoch): {val_accuracies_last_epoch}")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")
