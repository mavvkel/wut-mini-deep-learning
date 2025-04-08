import torch
from torch import nn, softmax

from cinic import get_loaders
from tree3_net import Tree3
from lenet import LeNet5

class MyEnsemble(nn.Module):
    def __init__(self, model_list, weights: list[float]):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)
        self.weights = weights
        self.weights_sum = sum(weights)

    def forward(self, x):
        return torch.stack([self.weights[i] * softmax(model(x), 1) for i, model in enumerate(self.model_list)]).sum(dim=0) / self.weights_sum


if __name__ == '__main__':
    batch_size = 32
    use_xpu = False

    # LeNet5s
    lenet_checkpoints = './lenet_checkpoints/'
    model1 = LeNet5()
    model1.load_state_dict(
        torch.load(
            './lenet_checkpoints/checkpoint_batch_size=64_weight_decay=1e-06_dropout_rate=0_learning_rate=0.0005_last.pth',
            map_location=lambda storage, _: storage 
        )['model_state_dict']
    )
    model1.eval()

    model2 = LeNet5()
    model2.load_state_dict(
        torch.load(
            './lenet_checkpoints/checkpoint_batch_size=64_weight_decay=1e-05_dropout_rate=0.1_learning_rate=0.001_last.pth',
            map_location=lambda storage, _: storage 
        )['model_state_dict']
    )
    model2.eval()

    model3 = LeNet5()
    model3.load_state_dict(
        torch.load(
            './lenet_checkpoints/checkpoint_batch_size=64_weight_decay=0.0001_dropout_rate=0.1_learning_rate=0.001_last.pth',
            map_location=lambda storage, _: storage 
        )['model_state_dict']
    )
    model3.eval()

    # Tree3s
    tree3_checkpoints = './checkpoints/'
    t_model1 = Tree3()
    t_model1.load_state_dict(
        torch.load(
            './checkpoints/aug-standard-32_w_decay-5e-05_momentum-0.96_try-1/aug_epochs_80.pt'
        )
    )
    t_model1.eval()

    t_model2 = Tree3()
    t_model2.load_state_dict(
        torch.load(
            #'./checkpoints/aug-flip-32_w_decay-5e-05_momentum-0.96_try-1/aug_epochs_40.pt'
            './checkpoints/aug-standard-32_w_decay-5e-05_momentum-0.96_try-2/aug_epochs_50.pt'
        )
    )
    t_model2.eval()

    t_model3 = Tree3()
    t_model3.load_state_dict(
        torch.load(
            # './checkpoints/aug-auto_standard-32_w_decay-5e-05_momentum-0.96_try-1/aug_epochs_100.pt'
            './checkpoints/aug-standard-32_w_decay-5e-05_momentum-0.96_try-3/aug_epochs_70.pt'
        )
    )
    t_model3.eval()

    ensemble = MyEnsemble(
        [model1, model2, model3, t_model1, t_model2, t_model3],
        # Weights correspond to individual accuracy of models
        weights=[0.5529, 0.5578, 0.5565, 0.6129, 0.6103, 0.6124],
        # weights=[1., 1., 1., 1., 1., 1],
    )

    _, _, testloader = get_loaders(batch_size)

    # Test accuracy
    correct = 0
    for _, (images, labels) in enumerate(testloader):
        if use_xpu:
            images = images.to('xpu')
            labels = labels.to('xpu')

        with torch.no_grad():
            outputs = ensemble(images)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

    test_accuracy = correct / 90000
    print('Test accuracy {}'.format(test_accuracy))

    with open("./test_accs.txt", "a") as f:
        f.write(f"Acc={test_accuracy} 3_lenet_3_tree_weighted_soft_voting\n")


