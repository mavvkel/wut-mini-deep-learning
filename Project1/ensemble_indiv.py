import torch
from torch import nn
from torchensemble.utils import io

from torchensemble import VotingClassifier

from tree3_net import Tree3
from cinic import get_loaders

EARLY_STOPING_PATIENCE = 10
EARLY_STOPING_DELTA = 0.00001
LR_PATIENCE = 5

epochs = 15
batch_size = 64
w_decay=0.00005
momentum=0.96

def train():
    trainloader, valloader, testloader = get_loaders(batch_size)

    # Define the ensemble
    ensemble = VotingClassifier(
        estimator=Tree3,               # here is your deep learning model
        n_estimators=10,                        # number of base estimators
        cuda=False,
    )
    # Set the criterion
    criterion = nn.CrossEntropyLoss()           # training objective
    ensemble.set_criterion(criterion)

    # Set the optimizer
    ensemble.set_optimizer(
        "SGD",                                 # type of parameter optimizer
        lr=0.1,                       # learning rate of parameter optimizer
        weight_decay=w_decay,              # weight decay of parameter optimizer
        momentum=momentum
    )

    ensemble.set_criterion(nn.CrossEntropyLoss())

    # Set the learning rate scheduler
    ensemble.set_scheduler(
        "ReduceLROnPlateau",                    # type of learning rate scheduler
        patience=LR_PATIENCE,                           # additional arguments on the scheduler
    )

    # Train the ensemble
    ensemble.fit(
        trainloader,
        epochs=epochs,                          # number of training epochs
        test_loader=valloader
    )

    # Evaluate the ensemble
    acc, loss = ensemble.evaluate(testloader, return_loss=True)

    with open("./test_accs.txt", "a") as f:
        f.write(f"Acc={acc} n_estimators{10} epochs={15} minibatch_size={batch_size}, w_decay={w_decay} momentum={momentum}\n")

    print(acc)
    print(loss)


if __name__ == '__main__':
    #ensemble = VotingClassifier(
    #    estimator=Tree3,               # here is your deep learning model
    #    n_estimators=10,                        # number of base estimators
    #    cuda=False,
    #)
    ## Set the criterion
    #criterion = nn.CrossEntropyLoss()           # training objective
    #ensemble.set_criterion(criterion)

    ## Set the optimizer
    #ensemble.set_optimizer(
    #    "SGD",                                 # type of parameter optimizer
    #    lr=0.1,                       # learning rate of parameter optimizer
    #    weight_decay=w_decay,              # weight decay of parameter optimizer
    #    momentum=momentum
    #)

    #ensemble.set_criterion(nn.CrossEntropyLoss())

    ## Set the learning rate scheduler
    #ensemble.set_scheduler(
    #    "ReduceLROnPlateau",                    # type of learning rate scheduler
    #    patience=LR_PATIENCE,                           # additional arguments on the scheduler
    #)

    #torch.serialization.add_safe_globals([CrossEntropyLoss])

    #io.load(ensemble, './')

    #trainloader, _, testloader = get_loaders(batch_size)
    #acc = ensemble.evaluate(testloader)
    #print(acc)

    train()
