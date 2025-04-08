import torch
from matplotlib import legend
import numpy as np
from statistics import stdev
import matplotlib.pyplot as plt
from numpy._core.multiarray import ndarray
from sklearn.metrics import confusion_matrix
import seaborn as sns

from cinic import get_loaders
from tree3_net import Tree3

def plot_batch_acc():
    batch_sizes = [32, 64, 128, 256]

    accs = {
        32: [0.5685333333333333, 0.5730555555555555, 0.5694666666666667],
        64: [0.5699777777777778, 0.5652888888888888, 0.5672555555555555],
        128: [0.49425555555555556, 0.5577555555555556, 0.5573888888888889],
        256: [0.5607333333333333, 0.5591777777777778, 0.5594444444444444],
    }

    avgs = {
        b_size: sum(acc_list) / 3
        for b_size, acc_list in accs.items()
    }

    stds = {
        b_size: stdev(acc_list)
        for b_size, acc_list in accs.items()
    }

    plt.figure(figsize=(15, 15))

    plt.title('Average accuracy vs. training batch size', fontsize=20)
    plt.xlabel('Batch size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.xticks([1, 2, 3, 4], [f"{s}" for s in batch_sizes])
    plt.ylim((0.49, 0.59))

    plt.scatter([1, 2, 3, 4], list(avgs.values()))

    plt.errorbar(
        [1, 2, 3, 4],
        list(avgs.values()),
        yerr=list(stds.values()),
        fmt='o',
        label='Average accuracy with 1 standard deviation',
        lw=1,
        ecolor='gray',
        capsize=10,
    )
    plt.legend(loc='lower right', fontsize=16)

    plt.show()

def plot_momentum_acc():
    momentums = [0.95, 0.96, 0.97, 0.98]

    accs = {
        0.95: [0.5639666666666666, 0.5626666666666666, 0.5597555555555556],
        0.96: [0.5685333333333333, 0.5730555555555555, 0.5694666666666667],
        0.97: [0.5601111111111111, 0.5597, 0.5580666666666667],
        0.98: [0.5669444444444445, 0.5654444444444444, 0.5648888888888889]
    }

    avgs = {
        b_size: sum(acc_list) / 3
        for b_size, acc_list in accs.items()
    }

    stds = {
        b_size: stdev(acc_list)
        for b_size, acc_list in accs.items()
    }

    plt.figure(figsize=(15, 15))

    plt.title('Average accuracy vs. momentum', fontsize=20)
    plt.xlabel('Momentum', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.ylim((0.555, 0.575))

    plt.scatter(momentums, list(avgs.values()))

    plt.errorbar(
        momentums,
        list(avgs.values()),
        yerr=list(stds.values()),
        fmt='o',
        label='Average accuracy with 1 standard deviation',
        lw=1,
        ecolor='gray',
        capsize=10,
    )
    plt.legend(loc='lower right', fontsize=16)

    plt.show()


def plot_wd_acc():
    decays = [
        0.0000005,
        0.000005,
        0.00005,
        0.0005,
    ]

    accs = {
        0.0000005: [0.4569888888888889, 0.4574111111111111, 0.4509888888888889],
        0.000005: [0.47396666666666665, 0.4502, 0.45265555555555553],
        0.00005:  [0.49425555555555556, 0.5577555555555556, 0.5573888888888889],
        0.0005:  [0.5452222222222223, 0.5463, 0.5490888888888888],
    }

    avgs = {
        dec: sum(acc_list) / 3
        for dec, acc_list in accs.items()
    }

    stds = {
        dec: stdev(acc_list)
        for dec, acc_list in accs.items()
    }

    plt.figure(figsize=(15, 15))

    plt.title('Average accuracy vs. weight decay', fontsize=20)
    plt.xlabel('Weight decay', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.xticks([1, 2, 3, 4], [f"{s}" for s in decays])
    plt.ylim((0.44, 0.59))

    plt.scatter([1, 2, 3, 4], list(avgs.values()))

    plt.errorbar(
        [1, 2, 3, 4],
        list(avgs.values()),
        yerr=list(stds.values()),
        fmt='o',
        label='Average accuracy with 1 standard deviation',
        lw=1,
        ecolor='gray',
        capsize=10,
    )
    plt.legend(loc='lower right', fontsize=16)

    plt.show()

def plot_quick_conv():
    s128 = np.genfromtxt('./plotting/Mar27_19-08-00_mavv2local-128_w_decay-5e-05_momentum-0.97_try-1-log_loss.csv', delimiter=',')
    s64  = np.genfromtxt('./plotting/Mar28_13-45-08_mavv2local-64_w_decay-5e-05_momentum-0.96_try-3-log_loss.csv', delimiter=',')
    s32 = np.genfromtxt('./plotting/Mar29_14-55-30_mavv2local-32_w_decay-5e-05_momentum-0.96_try-2-log_loss.csv', delimiter=',')

    plt.figure(figsize=(15, 10))

    plt.title('Average validation loss during training on strict early stopping parameters', fontsize=20, pad=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Validation loss', fontsize=16)
    plt.plot(s128[1:, 1], s128[1:, 2], label="Run with batch size = 128, weight decay = 5e-05, momentum = 0.97")
    plt.plot(s64[1:, 1], s64[1:, 2], label="Run with batch size = 64, weight decay = 5e-05, momentum = 0.96")
    plt.plot(s32[1:, 1], s32[1:, 2], label="Run with batch size = 32, weight decay = 5e-05, momentum = 0.96")
    plt.xlim(0, 40)
    plt.yscale('log')
    t = [0.4, 0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.008, 0.006]
    ts = [f"{tv}" for tv in t]

    def ann_pt(pt):
        plt.scatter(pt[0], pt[1])
        plt.annotate(
            f"({pt[0]:.0f}, {pt[1]:.4f})", xy=pt, xytext=(-3, 1), fontsize=14, textcoords='offset fontsize'
        )

    ann_pt((s128[-1, 1], s128[-1, 2]))
    ann_pt((s64[-1, 1], s64[-1, 2]))
    ann_pt((s32[-1, 1], s32[-1, 2]))

    plt.grid(True)
    plt.yticks(t, ts)
    plt.legend(fontsize=14)
    plt.show()


def plot_t3_stag():
    s128 = np.genfromtxt('./plotting/stag/Mar27_06-02-18_mavv2local-64_w_decay-5e-05_momentum-0.96.csv', delimiter=',')
    s64  = np.genfromtxt('./plotting/stag/Mar27_02-32-50_mavv2local-128_w_decay-5e-05_momentum-0.96.csv', delimiter=',')
    s32 = np.genfromtxt('./plotting/stag/Mar29_19-59-24_073ad6072b3fcollab-32_w_decay-0.0005_momentum-0.96_try-2.csv', delimiter=',')

    plt.figure(figsize=(15, 10))

    plt.title('Average validation loss during training without early stopping criterion', fontsize=20, pad=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Validation loss', fontsize=16)
    plt.plot(s128[1:, 1], s128[1:, 2], label="Run with batch size = 128, weight decay = 5e-05, momentum = 0.96")
    plt.plot(s64[1:, 1], s64[1:, 2], label="Run with batch size = 64, weight decay = 5e-05, momentum = 0.96")
    plt.plot(s32[1:, 1], s32[1:, 2], label="Run with batch size = 32, weight decay = 5e-04, momentum = 0.96")
    plt.xlim(0, 100)
    plt.yscale('log')
    t = [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    ts = [f"{tv}" for tv in t]

    def ann_pt(pt):
        plt.scatter(pt[0], pt[1])
        plt.annotate(
            f"({pt[0]:.0f}, {pt[1]:.4f})", xy=pt, xytext=(-3, 1), fontsize=14, textcoords='offset fontsize'
        )

    #ann_pt((s128[-1, 1], s128[-1, 2]))
    #ann_pt((s64[-1, 1], s64[-1, 2]))
    #ann_pt((s32[-1, 1], s32[-1, 2]))

    plt.grid(True)
    plt.yticks(t, ts)
    plt.legend(fontsize=14)
    plt.show()


def plot_t3_sched_lr():
    lr = np.genfromtxt('./plotting/sched_lr/Apr01_07-32-32_mavv2aug-autoaugment-32_w_decay-5e-05_momentum-0.96_try-3 lr.csv', delimiter=',')
    acc = np.genfromtxt('./plotting/sched_lr/Apr01_07-32-32_mavv2aug-autoaugment-32_w_decay-5e-05_momentum-0.96_try-3 (1).csv', delimiter=',')

    _, ax1 = plt.subplots(figsize=(15, 10))
    ax2 = ax1.twinx()


    plt.title('Example of accuracy & learning rate dynamics during training', fontsize=20, pad=20)

    plt.xlabel('Epochs', fontsize=16)
    plt.xlim(0, 70)

    l_acc = ax1.plot(acc[1:, 1], acc[1:, 2], 'b-', label="Accuracy")
    ax1.set_ylabel('Accuracy', fontsize=16)

    l_lr = ax2.plot(lr[1:, 1], lr[1:, 2], 'g-', label="Learning rate")
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning rate', fontsize=16)

    t_acc = [0.45, 0.5, 0.55, 0.6]
    ts_acc = [f"{tv}" for tv in t_acc]
    ax1.set_yticks(t_acc, ts_acc)

    t = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    ts = [f"{tv}" for tv in t]
    ax2.set_yticks(t, ts)
    lns = l_acc + l_lr
    # added these three lines
    ax2.legend(lns, ["Accuracy", "Learning rate"], loc="lower left", fontsize=14)

    #plt.legend(fontsize=14)

    plt.grid(axis='x')
    plt.yticks(t, ts)
    plt.show()

def plot_batch_stability():
    s32_l1 = np.genfromtxt('./plotting/stability/Mar29_13-24-11_mavv2local-32_w_decay-5e-05_momentum-0.96_try-1 loss.csv', delimiter=',')
    s32_l2 = np.genfromtxt('./plotting/stability/Mar29_14-55-30_mavv2local-32_w_decay-5e-05_momentum-0.96_try-2 loss.csv', delimiter=',')
    s32_l3 = np.genfromtxt('./plotting/stability/Mar29_16-02-32_mavv2local-32_w_decay-5e-05_momentum-0.96_try-3 loss.csv', delimiter=',')

    s64_l1 = np.genfromtxt('./plotting/stability/Mar27_06-02-18_mavv2local-64_w_decay-5e-05_momentum-0.96 loss.csv', delimiter=',')
    s64_l2 = np.genfromtxt('./plotting/stability/Mar27_19-25-27_2dea17f7da21collab-64_w_decay-5e-05_momentum-0.96_try-2 loss.csv', delimiter=',')
    s64_l3 = np.genfromtxt('./plotting/stability/Mar28_13-45-08_mavv2local-64_w_decay-5e-05_momentum-0.96_try-3 loss.csv', delimiter=',')

    s128_l1 = np.genfromtxt('./plotting/stability/Mar27_02-32-50_mavv2local-128_w_decay-5e-05_momentum-0.96 loss.csv', delimiter=',')
    s128_l2 = np.genfromtxt('./plotting/stability/Mar27_16-17-47_2dea17f7da21collab-128_w_decay-5e-05_momentum-0 loss.csv', delimiter=',')
    s128_l3 = np.genfromtxt('./plotting/stability/Mar28_08-59-28_mavv2local-128_w_decay-5e-05_momentum-0.96_try-3 loss.csv', delimiter=',')

    s256_l1 = np.genfromtxt('./plotting/stability/256 1 loss.csv', delimiter=',')
    s256_l2 = np.genfromtxt('./plotting/stability/256 2 loss.csv', delimiter=',')
    s256_l3 = np.genfromtxt('./plotting/stability/256 3 loss.csv', delimiter=',')

    plt.figure(figsize=(15, 10))

    plt.title('Average loss during training across different batch sizes', fontsize=20, pad=20)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Average loss', fontsize=16)

    def envel(l1: np.ndarray, l2: np.ndarray, l3: np.ndarray, size: str):
        t = [l1, l2, l3]
        t.sort(key=lambda arr: arr.shape[0], reverse=True)

        longest_val = np.array([l1.shape[0], l2.shape[0], l3.shape[0]]).max()

        mat = np.zeros((longest_val - 1, 3))
        mat[:, 0] = t[0][1:, 2]
        mat[:, 1] = t[0][1:, 2]
        mat[:, 2] = t[0][1:, 2]

        mat[0:len(t[1]) - 1, 1] = t[1][1:, 2]
        mat[0:len(t[2]) - 1, 2] = t[2][1:, 2]

        l_min = np.min(mat, axis=1)
        l_max = np.max(mat, axis=1)
        l_av = np.mean(mat, axis=1)

        longest_x = t[0][1:, 1]
        plt.plot(longest_x, l_av, label=f"Average loss for batch size s={size}")
        plt.fill_between(longest_x, l_min, l_max, alpha=0.2)

    envel(s32_l1, s32_l2, s32_l3, '32')
    envel(s64_l1, s64_l2, s64_l3, '64')
    envel(s128_l1, s128_l2, s128_l3, '128')
    envel(s256_l1, s256_l2, s256_l3, '256')

    plt.xlim((0, 50))
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.show()


def plot_augs():
    accs = {
        'horizontal flip': [0.6035888888888888, 0.6029, 0.6029],
        'rotation': [0.5895666666666667, 0.5894, 0.5896333333333333],
        'color jitter': [0.5755555555555556, 0.5760555555555555, 0.5751555555555555],
        '3 standard combined': [0.6129222222222223, 0.6102777777777778, 0.6124333333333334],
        'AutoAugment': [0.5889777777777778, 0.5850111111111111, 0.5853444444444444],
        'All combined': [0.5910888888888889, 0.5859777777777778, 0.5871444444444445],
    }

    avgs = {
        aug: sum(acc_list) / 3
        for aug, acc_list in accs.items()
    }

    stds = {
        aug: stdev(acc_list)
        for aug, acc_list in accs.items()
    }

    plt.figure(figsize=(15, 15))

    plt.title('Average accuracy obtained vs. augmentation strategy', fontsize=20, pad=15)
    plt.xlabel('Augmentation strategy', fontsize=16, labelpad=15)
    plt.ylabel('Accuracy', fontsize=16, labelpad=15)

    plt.xticks([1, 2, 3, 4, 5, 6], [s for s in list(accs.keys())])
    plt.ylim((0.57, 0.62))

    plt.scatter([1, 2, 3, 4, 5, 6], list(avgs.values()))

    plt.errorbar(
        [1, 2, 3, 4, 5, 6],
        list(avgs.values()),
        yerr=list(stds.values()),
        fmt='o',
        label='Average accuracy with 1 standard deviation',
        lw=1,
        ecolor='gray',
        capsize=10,
    )
    plt.legend(loc='lower right', fontsize=16)
    plt.tight_layout()

    plt.show()

def plot_conf_mat():
    # Load model
    t_model = Tree3()
    t_model.load_state_dict(
        torch.load(
            #'./checkpoints/aug-flip-32_w_decay-5e-05_momentum-0.96_try-1/aug_epochs_40.pt'
            './checkpoints/aug-standard-32_w_decay-5e-05_momentum-0.96_try-2/aug_epochs_50.pt'
        )
    )
    t_model.eval()

    batch_size = 1000
    _, _, testloader = get_loaders(batch_size)
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    all_labels = []
    all_preds = []

    for i, (images, labels) in enumerate(testloader):
        print((i + 1) * batch_size)
        with torch.no_grad():
            outputs = t_model(images)

            _, preds = torch.max(outputs, 1)

            all_labels = all_labels + labels.tolist()
            all_preds = all_preds + preds.tolist()

        # if i == 3:
        #     break

    model_results = {
        "conf_mat": confusion_matrix(all_labels, all_preds),
        "class_acc": (np.diag(confusion_matrix(all_labels, all_preds)) / np.sum(confusion_matrix(all_labels, all_preds), axis=1)),
    }

    plt.figure(figsize=(15, 15))

    sns.heatmap(
        model_results["conf_mat"],
        annot=True,
        fmt='d',
        cmap='pink',
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(r"Confusion matrix for run $s=32$, $d=5e-5$, $m=0.96$, $\delta=1e-6$, $p_\epsilon=10$", fontsize=20, pad=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    #plot_batch_acc()
    # plot_batch_stability()
    # plot_momentum_acc()
    #plot_wd_acc()
    #plot_quick_conv()
    # plot_t3_stag()
    #plot_t3_sched_lr()
    #plot_augs()
    plot_conf_mat()
