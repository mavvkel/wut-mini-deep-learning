import os
import torch
import torch.nn as nn
from early_stop import EarlyStopping
from tree3_net import Model
from torch.utils.tensorboard.writer import SummaryWriter
from torchsummary import summary
from cinic import get_augment_loaders, get_augment_transforms, get_loaders
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

EARLY_STOPING_PATIENCE = 10
EARLY_STOPING_DELTA = 0.00001
LR_PATIENCE = 5

def generate_run(
    num_epochs: int, batch_size: int, w_decay: float, momentum: float, config_try: int,
):
    seed = 17 + config_try

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    init_lr = 0.1

    run_prefix=f"local-{batch_size}_w_decay-{w_decay}_momentum-{momentum}_try-{config_try}"
    writer = SummaryWriter(comment=run_prefix)

    my_model = Model()
    summary(my_model,(3,32,32))
    criterion = nn.CrossEntropyLoss()

    trainloader, valloader, testloader = get_loaders(batch_size)

    optimizer = optim.SGD(my_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=w_decay, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=LR_PATIENCE)

    checkpoint_dir='checkpoints/' + run_prefix + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    early_stopping = EarlyStopping(patience=EARLY_STOPING_PATIENCE, delta=EARLY_STOPING_DELTA, verbose=True)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.
        for _, (images, labels) in enumerate(trainloader):
            #feedforward
            output = my_model(images)
            optimizer.zero_grad()
            loss = criterion(output, labels)

            with torch.no_grad():
                epoch_train_loss += loss

            #backpropragation
            loss.backward()

            #update the weights/parameters
            optimizer.step()

        if epoch % 10 == 9:
            torch.save(my_model.state_dict(), checkpoint_dir + f"epochs_{epoch+1}.pt")

        # Train accuracy
        total = 0
        correct = 0
        epoch_valid_loss = 0.
        for _, (images, labels) in enumerate(valloader):
            with torch.no_grad():
                outputs = my_model(images)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                total += labels.size(0)

                epoch_valid_loss += criterion(outputs, labels)

        val_accuracy = correct / total

        av_epoch_train_loss = epoch_train_loss / 90000
        av_epoch_valid_loss = epoch_valid_loss / 90000

        print('Epoch {}: Val accuracy {:.6f}, Val loss {:.8f}'.format(epoch, val_accuracy, av_epoch_valid_loss))

        writer.add_scalar("Loss/train", av_epoch_train_loss, epoch)
        writer.add_scalar("Loss/valid", av_epoch_valid_loss, epoch)

        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch)

        # Check early stopping condition
        early_stopping.check_early_stop(av_epoch_valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")

            if epoch % 10 != 9:
                torch.save(my_model.state_dict(), checkpoint_dir + f"epochs_{epoch+1}.pt")
            break

        scheduler.step(av_epoch_valid_loss)

    writer.flush()

    # Test accuracy
    total = 0
    correct = 0

    for _, (images, labels) in enumerate(testloader):
        with torch.no_grad():
            outputs = my_model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

    test_accuracy = float(correct)/total
    print('Test accuracy {}' .format(test_accuracy))

    with open("./test_accs.txt", "a") as f:
        f.write(f"Acc={test_accuracy} minibatch_size={batch_size}, w_decay={w_decay} momentum={momentum} try={config_try}\n")


def generate_collab_run(
    num_epochs: int, batch_size: int, w_decay: float, momentum: float, config_try: int,
):
    seed = 17 + config_try

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    use_cuda = torch.cuda.is_available()
    print('Use GPU?', use_cuda)

    init_lr = 0.1

    run_prefix=f"collab-{batch_size}_w_decay-{w_decay}_momentum-{momentum}_try-{config_try}"
    writer = SummaryWriter(comment=run_prefix)

    my_model = Model()

    if use_cuda:
      my_model = my_model.cuda()

    summary(my_model,(3,32,32))
    criterion = nn.CrossEntropyLoss()

    trainloader, valloader, testloader = get_loaders(batch_size)

    my_model = Model()

    if use_cuda:
      my_model = my_model.cuda()

    optimizer = optim.SGD(my_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=w_decay, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=LR_PATIENCE)

    checkpoint_dir = 'checkpoints'
    checkpoint_subdir = 'checkpoints/'+ run_prefix

    if not os.path.exists(checkpoint_subdir):
        os.mkdir(checkpoint_dir)
        os.mkdir(checkpoint_subdir)

    early_stopping = EarlyStopping(patience=EARLY_STOPING_PATIENCE, delta=EARLY_STOPING_DELTA, verbose=True)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.
        for _, (images, labels) in enumerate(trainloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            #feedforward
            output = my_model(images)
            optimizer.zero_grad()
            loss = criterion(output, labels)

            with torch.no_grad():
                epoch_train_loss += loss

            #backpropragation
            loss.backward()

            #update the weights/parameters
            optimizer.step()

        if epoch % 10 == 9:
            torch.save(my_model.state_dict(), checkpoint_subdir + f"/epochs_{epoch+1}.pt")

        # Train accuracy
        total = 0
        correct = 0
        epoch_valid_loss = 0.
        for _, (images, labels) in enumerate(valloader):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                outputs = my_model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

            with torch.no_grad():
                epoch_valid_loss += criterion(outputs, labels)

        val_accuracy = float(correct) / total

        av_epoch_train_loss = epoch_train_loss / 90000
        av_epoch_valid_loss = epoch_valid_loss / 90000

        print('Epoch {}: Val accuracy {:.6f}, Val loss {:.8f}'.format(epoch, val_accuracy, av_epoch_valid_loss))

        writer.add_scalar("Loss/train", av_epoch_train_loss, epoch)
        writer.add_scalar("Loss/valid", av_epoch_valid_loss, epoch)

        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch)

        # Check early stopping condition
        early_stopping.check_early_stop(av_epoch_valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")

            if epoch % 10 != 9:
                torch.save(my_model.state_dict(), checkpoint_dir + f"epochs_{epoch+1}.pt")
            break

        scheduler.step(av_epoch_valid_loss)

    writer.flush()

    # Test accuracy
    total = 0
    correct = 0

    for _, (images, labels) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            outputs = my_model(images)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

    test_accuracy = float(correct)/total
    print('Test accuracy {}' .format(test_accuracy))

    with open("./test_accs.txt", "a") as f:
        f.write(f"Acc={test_accuracy} minibatch_size={batch_size}, w_decay={w_decay} momentum={momentum} try={config_try}\n")


def generate_augmented_run(
    num_epochs: int,
    batch_size: int,
    w_decay: float,
    momentum: float,
    config_try: int,
    transform: str,
    checkpoint_path: str,
):
    seed = 17 + config_try

    torch.manual_seed(seed)
    torch.xpu.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

    use_xpu = False
    print('Use xGPU?', use_xpu)

    init_lr = 0.1

    run_prefix=f"aug-{transform}-{batch_size}_w_decay-{w_decay}_momentum-{momentum}_try-{config_try}"
    writer = SummaryWriter(comment=run_prefix)

    print("Running " + run_prefix)

    criterion = nn.CrossEntropyLoss()

    aug_transforms = get_augment_transforms()
    trainloader, valloader, testloader = get_augment_loaders(batch_size, aug_transforms[transform])

    my_model = Model()
    my_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    if use_xpu:
      my_model = my_model.to('xpu')

    optimizer = optim.SGD(my_model.parameters(), lr=init_lr, momentum=momentum, weight_decay=w_decay, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=LR_PATIENCE)

    checkpoint_dir = 'checkpoints'
    checkpoint_subdir = 'checkpoints/'+ run_prefix

    if not os.path.exists(checkpoint_subdir):
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        os.mkdir(checkpoint_subdir)

    early_stopping = EarlyStopping(patience=EARLY_STOPING_PATIENCE, delta=EARLY_STOPING_DELTA, verbose=True)

    for epoch in range(num_epochs):
        epoch_train_loss = 0.
        for _, (images, labels) in enumerate(trainloader):
            if use_xpu:
                images = images.to('xpu')
                labels = labels.to('xpu')

            #feedforward
            output = my_model(images)
            optimizer.zero_grad()
            loss = criterion(output, labels)

            with torch.no_grad():
                epoch_train_loss += loss

            #backpropragation
            loss.backward()

            #update the weights/parameters
            optimizer.step()

        if epoch % 10 == 9:
            torch.save(my_model.state_dict(), checkpoint_subdir + f"/aug_epochs_{epoch+1}.pt")

        # Train accuracy
        total = 0
        correct = 0
        epoch_valid_loss = 0.
        for _, (images, labels) in enumerate(valloader):
            if use_xpu:
                images = images.to('xpu')
                labels = labels.to('xpu')

            with torch.no_grad():
                outputs = my_model(images)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

            with torch.no_grad():
                epoch_valid_loss += criterion(outputs, labels)

        val_accuracy = float(correct) / total

        av_epoch_train_loss = epoch_train_loss / 90000
        av_epoch_valid_loss = epoch_valid_loss / 90000

        print('Epoch {}: Val accuracy {:.6f}, Val loss {:.8f}'.format(epoch, val_accuracy, av_epoch_valid_loss))

        writer.add_scalar("Loss/train", av_epoch_train_loss, epoch)
        writer.add_scalar("Loss/valid", av_epoch_valid_loss, epoch)

        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[-1], epoch)

        # Check early stopping condition
        early_stopping.check_early_stop(av_epoch_valid_loss)

        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch}")

            if epoch % 10 != 9:
                torch.save(my_model.state_dict(), checkpoint_dir + f"/aug_epochs_{epoch+1}.pt")
            break

        scheduler.step(av_epoch_valid_loss)

    writer.flush()

    # Test accuracy
    total = 0
    correct = 0

    for _, (images, labels) in enumerate(testloader):
        if use_xpu:
            images = images.to('xpu')
            labels = labels.to('xpu')

        with torch.no_grad():
            outputs = my_model(images)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

    test_accuracy = float(correct)/total
    print('Test accuracy {}' .format(test_accuracy))

    with open("./test_accs.txt", "a") as f:
        f.write(f"Acc={test_accuracy} aug={transform} minibatch_size={batch_size}, w_decay={w_decay} momentum={momentum} try={config_try}\n")


if __name__ == '__main__':
    generate_run(
        num_epochs=2,
        batch_size=32,
        w_decay=0.00005,
        momentum=0.96,
        config_try=1,
    )
