import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import defaultdict
import random
import os
import time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from easyfsl.methods import PrototypicalNetworks
from PIL import ImageEnhance
from PIL import Image, ImageFilter
from lenet.py import LeNet5
from early_stop.py import EarlyStopping



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
  
mean = [0.4605, 0.4605, 0.4605]
std = [0.2475, 0.2475, 0.2475]

transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

class EasyFSLImageFolder(datasets.ImageFolder):
    def get_labels(self):
        return [label for _, label in self.samples]

transform_test = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = EasyFSLImageFolder("archive/train", transform=transform)
valid_dataset = EasyFSLImageFolder("archive/valid", transform=transform_test)
test_dataset = EasyFSLImageFolder("archive/test", transform=transform_test)

n_way, n_shot, n_query, n_tasks = 10, 5, 15, 400

class FewShotDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

    def get_task(self, n_way, n_shot, n_query):
        selected_classes = random.sample(self.classes, n_way)

        support_indices = []
        query_indices = []

        for cls in selected_classes:
            indices = random.sample(self.class_to_indices[cls], n_shot + n_query)
            support_indices += indices[:n_shot]
            query_indices += indices[n_shot:]

        support_images, support_labels = zip(*[self.dataset[i] for i in support_indices])
        query_images, query_labels = zip(*[self.dataset[i] for i in query_indices])

        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)

        label_map = {cls: i for i, cls in enumerate(selected_classes)}
        support_labels = torch.tensor([label_map[label] for label in support_labels])
        query_labels = torch.tensor([label_map[label] for label in query_labels])

        return support_images, support_labels, query_images, query_labels

train_fs = FewShotDatasetWrapper(train_dataset)
val_fs = FewShotDatasetWrapper(valid_dataset)

def few_shot_loader(fsdataset, n_tasks):
    for _ in range(n_tasks):
        yield fsdataset.get_task(n_way=n_way, n_shot=n_shot, n_query=n_query)

train_loader = few_shot_loader(train_fs, n_tasks=n_tasks)
val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)

def extract_support_query(task):
    support_images, support_labels, query_images, query_labels = task
    return support_images.to(device), support_labels.to(device), query_images.to(device), query_labels.to(device)

class LeNet5Backbone(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout1(self.relu3(self.fc1(x)))
        x = self.dropout2(self.relu4(self.fc2(x)))
        return x

backbone = LeNet5Backbone(dropout_rate=0.3).to(device)
model = PrototypicalNetworks(backbone).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter("runs/fewshot_protonet")

epochs = 100
best_val_loss = float("inf")
early_stop_counter = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    train_loader = few_shot_loader(train_fs, n_tasks=n_tasks)  

    for task in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training Tasks"):
        support_images, support_labels, query_images, query_labels = extract_support_query(task)

        model.process_support_set(support_images, support_labels)
        outputs = model(query_images)
        outputs = torch.nan_to_num(outputs)

        loss = criterion(outputs, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == query_labels).sum().item()
        total += query_labels.size(0)

    avg_train_loss = running_loss / 100  
    train_acc = 100 * correct / total if total > 0 else 0.0
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)

    with torch.no_grad():
        for task in tqdm(val_loader, desc="Validation Tasks"):
            support_images, support_labels, query_images, query_labels = extract_support_query(task)

            model.process_support_set(support_images, support_labels)
            outputs = model(query_images)
            outputs = torch.nan_to_num(outputs)

            val_loss += criterion(outputs, query_labels).item()
            val_correct += (outputs.argmax(1) == query_labels).sum().item()
            val_total += query_labels.size(0)

    avg_val_loss = val_loss / 100
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_protonet.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping triggered!")
            break

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.load_state_dict(torch.load("best_protonet.pth"))
model.eval()
all_preds, all_targets = [], []

val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)

with torch.no_grad():
    for task in tqdm(val_loader, desc="Final Evaluation"):
        support_images, support_labels, query_images, query_labels = extract_support_query(task)

        model.process_support_set(support_images, support_labels)
        outputs = model(query_images)
        outputs = torch.nan_to_num(outputs)
        pred = outputs.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(query_labels.cpu().numpy())

cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])


transform_test = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = EasyFSLImageFolder("archive/train", transform=transform)
valid_dataset = EasyFSLImageFolder("archive/valid", transform=transform_test)
test_dataset = EasyFSLImageFolder("archive/test", transform=transform_test)

n_way, n_shot, n_query, n_tasks = 10, 5, 15, 100

train_fs = FewShotDatasetWrapper(train_dataset)
val_fs = FewShotDatasetWrapper(valid_dataset)


train_loader = few_shot_loader(train_fs, n_tasks=n_tasks)
val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)


class TreeLayer(nn.Module):
    def __init__(self, sizes, activation):
        super().__init__()
        self.sizes = sizes
        self.activation = activation

        weights = torch.empty(self.sizes)
        bias = torch.empty(1, self.sizes[1], self.sizes[3], self.sizes[4])
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(bias)

        if self.activation == "relu":
            nn.init.kaiming_normal_(self.weights)
        elif self.activation == "sigmoid":
            nn.init.normal_(self.weights, mean=0.0, std=1.0)
        nn.init.kaiming_normal_(self.bias)

    def forward(self, x):
        w_times_x = torch.mul(x, self.weights)
        w_times_x = torch.sum(w_times_x, dim=[2,5,6,7])
        w_times_x = torch.add(w_times_x, self.bias)
        return w_times_x

class TreeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 3
        self.num_filters_conv1 = 15
        self.num_filters_conv2 = 16
        self.activation = "sigmoid"

        self.conv1 = nn.Conv2d(3, self.num_filters_conv1 * self.num_groups, kernel_size=5, groups=self.num_groups)
        self.tree1 = TreeLayer((1,self.num_filters_conv2,self.num_filters_conv1,self.num_groups,7,7,2,2), self.activation)
        self.output_dim = 7 * self.num_filters_conv2 * 3

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.unfold(x, (2, 2), stride=2)
        x = x.reshape(-1, 1 * self.num_filters_conv1 * self.num_groups, 2 * 2, 7 * 7).transpose(2, 3)
        x = x.reshape(-1, self.num_groups, 1, self.num_filters_conv1, 7, 7, 2, 2).transpose(1, 2).transpose(2, 3)
        x = torch.sigmoid(self.tree1(x))
        return x.reshape(x.size(0), -1)


EARLY_STOPING_PATIENCE = 10
EARLY_STOPING_DELTA = 0.00001
backbone = TreeBackbone().to(device)
model = PrototypicalNetworks(backbone).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.96, weight_decay=0.00005, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
early_stopping = EarlyStopping(patience=EARLY_STOPING_PATIENCE, delta=EARLY_STOPING_DELTA, verbose=True)

epochs = 100
best_val_loss = float("inf")
early_stop_counter = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    train_loader = few_shot_loader(train_fs, n_tasks=n_tasks)  

    for task in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training Tasks"):
        support_images, support_labels, query_images, query_labels = extract_support_query(task)

        model.process_support_set(support_images, support_labels)
        outputs = model(query_images)
        outputs = torch.nan_to_num(outputs)

        loss = criterion(outputs, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == query_labels).sum().item()
        total += query_labels.size(0)

    avg_train_loss = running_loss / n_tasks 
    train_acc = 100 * correct / total if total > 0 else 0.0
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)

    with torch.no_grad():
        for task in tqdm(val_loader, desc="Validation Tasks"):
            support_images, support_labels, query_images, query_labels = extract_support_query(task)

            model.process_support_set(support_images, support_labels)
            outputs = model(query_images)
            outputs = torch.nan_to_num(outputs)

            val_loss += criterion(outputs, query_labels).item()
            val_correct += (outputs.argmax(1) == query_labels).sum().item()
            val_total += query_labels.size(0)

    avg_val_loss = val_loss / n_tasks
    val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Val", val_acc, epoch)

    print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_tree_protonet.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping triggered!")
            break


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()


model.load_state_dict(torch.load("best_tree_protonet.pth"))
model.eval()
all_preds, all_targets = [], []

val_loader = few_shot_loader(val_fs, n_tasks=n_tasks)

with torch.no_grad():
    for task in tqdm(val_loader, desc="Final Evaluation"):
        support_images, support_labels, query_images, query_labels = extract_support_query(task)

        model.process_support_set(support_images, support_labels)
        outputs = model(query_images)
        outputs = torch.nan_to_num(outputs)
        pred = outputs.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(query_labels.cpu().numpy())


cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
