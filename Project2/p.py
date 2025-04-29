from math import nan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
import torch
from ASR import ASRModel
from AST import ASTModel
from dataset import CLASS_NAMES_TO_IXS, get_dataloaders
from resnet18 import ResNet, resnet18
import seaborn as sns
from sklearn.metrics import confusion_matrix

def ann_pt(pt, ax, fs, xytext, c=None):
    if c is None:
        ax.scatter(pt[0], pt[1])
    else:
        ax.scatter(pt[0], pt[1], color=c)
    ax.annotate(
        f"({pt[0]:.0f}, {pt[1]:.4f})", xy=pt, xytext=xytext, fontsize=fs, textcoords='offset fontsize'
    )

#def _acc_loss_plot(vacs, tacs, vlos, tlos):
#    fig, axs = plt.subplots(2, 2, figsize=[14, 14])
#
#    for tac, lab in vacs.items():
#        x = tac[1:, 1]
#        y = tac[1:, 2]
#        axs[0, 0].plot(x, y, color='tab:orange')
#
#    for vac, lab in vacs.items():
#        x = vac[1:, 1]
#        y = vac[1:, 2]
#        axs[0, 0].plot(x, y, color='tab:orange')
#
#    axs[0, 0].plot(x, y)
#    axs[0, 0].set_title('Axis [0, 0]')
#    axs[0, 1].plot(x, y, 'tab:orange')
#    axs[0, 1].set_title('Axis [0, 1]')
#    axs[1, 0].plot(x, -y, 'tab:green')
#    axs[1, 0].set_title('Axis [1, 0]')
#    axs[1, 1].plot(x, -y, 'tab:red')
#    axs[1, 1].set_title('Axis [1, 1]')
#
#    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=[12, 12])
#
#    for tac, lab in tacs.items():
#        x = tac[1:, 1]
#        y = tac[1:, 2]
#        ax1.plot(x, y, color='tab:orange')


def rms_best():
    fs=18
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_tac, y_tac = ld('./rms_best/tac_Apr26_20-10-44_fd175db88ada__ResNet__RMSprop__(32,0.0001,0.0001).csv')
    x_vac, y_vac = ld('./rms_best/vac_Apr26_20-10-44_fd175db88ada__ResNet__RMSprop__(32,0.0001,0.0001).csv')
    x_tlo, y_tlo = ld('./rms_best/tlo_Apr26_20-10-44_fd175db88ada__ResNet__RMSprop__(32,0.0001,0.0001).csv')
    x_vlo, y_vlo = ld('./rms_best/vlo_Apr26_20-10-44_fd175db88ada__ResNet__RMSprop__(32,0.0001,0.0001).csv')

    fig.suptitle(r'Training & validation loss and accuracy for $(s, \eta, d) = (32, 0.0001, 0.0001)$', fontsize=fs)
    ax1.plot(x_tac, y_tac, color='tab:blue', label='Training accuracy')
    ax1.plot(x_vac, y_vac, color='tab:orange', label='Validation accuracy')
    ax1.legend(fontsize=fs)
    ax1.set_xticks([i for i in range(0, 21, 2)])

    ann_pt((x_tac[-1], y_tac[-1]), ax1, fs=12, xytext=(-5.5, 0.5))
    ann_pt((x_vac[-1], y_vac[-1]), ax1, fs=12, xytext=(-5.5, 0.5))

    ax2.plot(x_tlo, y_tlo, color='tab:blue', label='Traing loss')
    ax2.plot(x_vlo, y_vlo, color='tab:orange', label='Validation loss')

    ax2.set_xticks([i for i in range(0, 21, 2)])
    ax2.legend(fontsize=fs)

    plt.show()


def resnet_stag():
    fs=18
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_tac, y_tac = ld('./resnet_stag/tac_Apr27_00-09-40_mavv2__ResNet__Adam__(32,0.0005,0).csv')
    x_vac, y_vac = ld('./resnet_stag/vac_Apr27_00-09-40_mavv2__ResNet__Adam__(32,0.0005,0).csv')
    x_tlo, y_tlo = ld('./resnet_stag/tlo_Apr27_00-09-40_mavv2__ResNet__Adam__(32,0.0005,0).csv')
    x_vlo, y_vlo = ld('./resnet_stag/vlo_Apr27_00-09-40_mavv2__ResNet__Adam__(32,0.0005,0).csv')

    fig.suptitle(r'Training & validation loss and accuracy for $(s, \eta, d) = (32, 0.0005, 9)$', fontsize=fs)
    ax1.plot(x_tac, y_tac, color='tab:blue', label='Training accuracy')
    ax1.plot(x_vac, y_vac, color='tab:red', label='Validation accuracy')
    ax1.legend(fontsize=fs)
    ax1.set_xticks([i for i in range(0, 21, 2)])

    ann_pt((x_tac[-1], y_tac[-1]), ax1, fs=12, xytext=(-5.5, 0.5), c='tab:blue')
    ann_pt((x_vac[-1], y_vac[-1]), ax1, fs=12, xytext=(-5.5, 0.5), c='tab:red')

    ax2.plot(x_tlo, y_tlo, color='tab:blue', label='Traing loss')
    ax2.plot(x_vlo, y_vlo, color='tab:orange', label='Validation loss')

    ax2.set_xticks([i for i in range(0, 21, 2)])
    ax2.legend(fontsize=fs)

    plt.show()

def _3d_scat(s32, s64, s128):

    fig, axs = plt.subplots(1,3,subplot_kw=dict(projection='3d'))

    sizes = [32, 64, 128]
    for i, (s, c) in enumerate([(s32, 'tab:blue'), (s64, 'tab:orange'), (s128, 'tab:green')]):
        ax = axs[i]
        s = np.asarray(s)
        # x=lr, y=d, z=acc
        x = s[:, 1]
        x[x==0.001] = 1
        x[x==0.0005] = 2
        x[x==0.0001] = 3

        y = s[:, 2]
        y[y==0] = 1
        y[y==0.0001] = 2
        y[y==0.001] = 3

        z = s[:, 0]

        ax.plot_surface(
            x.reshape((3, 3)),
            y.reshape((3, 3)),
            z.reshape(3, 3),
            cstride=1,
            rstride=1,
            shade=False,
            color=c,
            alpha=0.5,
            edgecolor='black',
            lw=1
        )

        ax.scatter(x, y, s[:, 0], color=c, marker='o')
        #ms, _, bs = ax.stem(x, y, s[:, 0])
        #bs.set_color

        ax.view_init(elev=10, azim=-135, roll=0)
        ax.set_xticks([1, 2, 3], ['0.001', '0.0005', '0.0001'])
        ax.set_yticks([1, 2, 3], ['0', '0.0001', '0.001'])
        ax.set_zlim((0.4, 0.9))

        ax.set_title(f"Batch size $s={sizes[i]}$", fontsize=16, pad=0)
        ax.set_xlabel(r'Learning rate $\eta$')
        ax.set_ylabel('Weight decay $d$')
        ax.set_zlabel('Test accuracy')

    fig.suptitle(r'Test accuracy against learning rate $\eta$ and weight decay $d$ for different values of batch size $s$', fontsize=20)

    plt.show()


def ld(path: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.genfromtxt(path, delimiter=',')

    return d[1:, 1], d[1:, 2]


#s32 = [
#    [0.46940649753185626, 0.001, 0],
#    [0.5576856847663874, 0.0005, 0],
#    [0.8321662266100333, 0.0001, 0],
#    [0.47158764780163015, 0.001, 0.0001],
#    [0.5599816324187809, 0.0005, 0.0001],
#    [0.8297554815750201, 0.0001, 0.0001],
#    [0.5108483526575595, 0.001, 0.001],
#    [0.5901733440477557, 0.0005, 0.001],
#    [0.7504304901848238, 0.0001, 0.001],
#]

#s64 = [
#    [0.5587188612099644, 0.001, 0],
#    [0.5935024681437263, 0.0005, 0],
#    [0.8225232464699804, 0.0001, 0],
#    [0.49902422224773274, 0.001, 0.0001],
#    [0.561359201010217, 0.0005, 0.0001],
#    [0.857881,           0.0001, 0.0001],
#    [0.5251980254850189, 0.001, 0.001],
#    [0.5891401676041786, 0.0005, 0.001],
#    [0.8438755596372403, 0.0001, 0.001],
#]

#s128 = [
#    [0.597520376535415, 0.001, 0],
#    [0.6587073814717025, 0.0005, 0],
#    [0.8257375731833314, 0.0001, 0],
#    [0.4838709677419355, 0.001, 0.0001],
#    [0.6637584663069682, 0.0005, 0.0001],
#    [0.796464,           0.0001, 0.0001],
#    [0.6315004017908392, 0.001, 0.001],
#    [0.6099418749418919, 0.0005, 0.001],
#    [0.8466306968201125, 0.0001, 0.001],
#]

def plot_conf_mat(model):
    # Load model
    model.eval()

    batch_size = 1000
    _, _, testloader = get_dataloaders(batch_size)

    #class_names = [
    #    "silence",
    #    "command",
    #    "unknown",
    #]
    class_names = [
        "down",
        "eight",
        "five",
        "four",
        "go",
        "left",
        "nine",
        "no",
        "off",
        "on",
        "one",
        "right",
        "seven",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "yes",
        "zero",
        "unknown",
        "silence"
    ]

    all_labels = []
    all_preds = []

    total = 0
    correct = 0
    for i, (waveforms, labels, _) in enumerate(testloader):
        print((i + 1) * batch_size)
        with torch.no_grad():
            if model.__class__.__name__ in [ASRModel.__name__, ASTModel.__name__]:
                waveforms = waveforms.squeeze(1)
            outputs = model(waveforms)

            _, preds = torch.max(outputs, 1)

            all_labels = all_labels + labels.tolist()
            all_preds = all_preds + preds.tolist()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

    test_accuracy = float(correct) / total
    print(f"\tAccuracy={test_accuracy:.6f}")
    model_results = {
        "conf_mat": confusion_matrix(all_labels, all_preds),
        "class_acc": (np.diag(confusion_matrix(all_labels, all_preds)) / np.sum(confusion_matrix(all_labels, all_preds), axis=1)),
    }

    plt.figure(figsize=(10, 10))

    #sns.set(font_scale=1.7)
    sns.heatmap(
        model_results["conf_mat"],
        annot=True,
        fmt='d',
        cmap='pink',
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(r"Confusion matrix for EST run with $(s, \eta, d) = (128, 0.0005, 0.0001)$", fontsize=20, pad=20)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.tight_layout()
    plt.show()

def ast_tonan():
    fs=16
    fig, (ax1, ax2) = plt.subplots(1, 2)

    vlos = {
        './plots/ast_tonan/vlo_Apr27_23-27-15_b26d4672e240__ASTModel__RMSprop__(32,0.001,0.001).csv': r'Run $s=32, \eta=0.001, d=0.001$',
        './plots/ast_tonan/vlo_Apr27_23-40-49_b26d4672e240__ASTModel__RMSprop__(32,0.0005,0.001).csv': r'Run $s=32, \eta=0.001, d=0.001$',
        './plots/ast_tonan/vlo_Apr28_00-47-12_b26d4672e240__ASTModel__RMSprop__(32,0.0005,0.0001).csv': r'Run $s=32, \eta=0.001, d=0.001$',
    }

    tlos = {
        './plots/ast_tonan/tlo_Apr27_23-27-15_b26d4672e240__ASTModel__RMSprop__(32,0.001,0.001).csv': r'Run $s=32, \eta=0.001, d=0.001$',
        './plots/ast_tonan/tlo_Apr27_23-40-49_b26d4672e240__ASTModel__RMSprop__(32,0.0005,0.001).csv': r'Run $s=32, \eta=0.0005, d=0.001$',
        './plots/ast_tonan/tlo_Apr28_00-47-12_b26d4672e240__ASTModel__RMSprop__(32,0.0005,0.0001).csv': r'Run $s=32, \eta=0.0005, d=0.0001$'
    }

    fig.suptitle(r'Metrics on degrading training runs on AST using $\text{RMSprop}$', fontsize=fs)
    for tlo, lab in tlos.items():
        x_tlo, y_tlo = ld(tlo)

        x_nan = x_tlo[np.isnan(y_tlo)]
        y_nan = np.full_like(x_nan, fill_value=y_tlo[np.isnan(y_tlo) == False][-1])
        ax1.scatter(x_nan, y_nan, marker='^', s=80)
        ax1.set_title('Training loss', fontsize=fs, pad=10)
        if len(y_tlo[np.isnan(y_tlo) == False]) == 1:
            ax1.scatter(x_tlo, y_tlo, marker='o', s=80, c='tab:blue')
        ax1.plot(x_tlo, y_tlo, label=lab)
        ax1.legend(fontsize=fs, loc='lower left')
        ax1.set_xticks([i for i in range(1, 11)])

    for vlo, lab in vlos.items():
        x_vlo, y_vlo = ld(vlo)

        x_nan = x_vlo[np.isnan(y_vlo)]
        y_nan = np.full_like(x_nan, fill_value=y_vlo[np.isnan(y_vlo) == False][-1])
        ax2.scatter(x_nan, y_nan, marker='^', s=80)

        ax2.set_title('Validation loss', fontsize=fs, pad=10)
        if len(y_vlo[np.isnan(y_vlo) == False]) == 1:
            ax2.scatter(x_vlo, y_vlo, marker='o', s=80, c='tab:blue')
        ax2.plot(x_vlo, y_vlo, label=lab)
        ax2.legend(fontsize=fs, loc='center right')
        ax2.set_xticks([i for i in range(1, 11)])

    plt.tight_layout()
    plt.show()

def ast_rms_best():
    fs=18
    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_tac, y_tac = ld('./plots/ast_rms_best/tac_Apr28_02-54-59_b26d4672e240__ASTModel__RMSprop__(64,0.0001,0.001).csv')
    x_vac, y_vac = ld('./plots/ast_rms_best/vac_Apr28_02-54-59_b26d4672e240__ASTModel__RMSprop__(64,0.0001,0.001).csv')
    x_tlo, y_tlo = ld('./plots/ast_rms_best/tlo_Apr28_02-54-59_b26d4672e240__ASTModel__RMSprop__(64,0.0001,0.001).csv')
    x_vlo, y_vlo = ld('./plots/ast_rms_best/vlo_Apr28_02-54-59_b26d4672e240__ASTModel__RMSprop__(64,0.0001,0.001).csv')

    fig.suptitle(r'Training & validation loss and accuracy for $(s, \eta, d) = (64, 0.0001, 0.001)$', fontsize=fs)
    ax1.plot(x_tac, y_tac, color='tab:blue', label='Training accuracy')
    ax1.plot(x_vac, y_vac, color='tab:orange', label='Validation accuracy')
    ax1.legend(fontsize=fs)
    ax1.set_xticks([i for i in range(0, 21, 2)])

    ann_pt((x_tac[-1], y_tac[-1]), ax1, fs=12, xytext=(-5.5, 0.5))
    ann_pt((x_vac[-1], y_vac[-1]), ax1, fs=12, xytext=(-5.5, 0.5))

    ax2.plot(x_tlo, y_tlo, color='tab:blue', label='Traing loss')
    ax2.plot(x_vlo, y_vlo, color='tab:orange', label='Validation loss')

    ax2.set_xticks([i for i in range(0, 21, 2)])
    ax2.legend(fontsize=fs)

    plt.show()

if __name__ == '__main__':
    #s32 = [
    #    [0.4494317529560326, 0.001, 0],
    #    [0.5882217885432213, 0.0005, 0],
    #    [0.836528527149581, 0.0001, 0],
    #    [0.4409367466421766, 0.001, 0.0001],
    #    [0.45459763517391805, 0.0005, 0.0001],
    #    [0.8829066697279302, 0.0001, 0.0001],
    #    [0.5313970841464815, 0.001, 0.001],
    #    [0.589714154517277, 0.0005, 0.001],
    #    [0.6907358512225922, 0.0001, 0.001],
    #]

    #s64 = [
    #    [0.5193433589714155, 0.001, 0],
    #    [0.5280679600505108, 0.0005, 0],
    #    [0.7413614969578693, 0.0001, 0],
    #    [0.5526345999311215, 0.001, 0.0001],
    #    [0.5264607966938354, 0.0005, 0.0001],
    #    [0.8682126047526116, 0.0001, 0.0001],
    #    [0.6160027551371828, 0.001, 0.001],
    #    [0.6751234071863161, 0.0005, 0.001],
    #    [0.7551371828722305, 0.0001, 0.001],
    #]

    #s128 = [
    #    [0.5789232005510274, 0.001, 0],
    #    [0.676960165308231, 0.0005, 0],
    #    [0.8459419125243944, 0.0001, 0],
    #    [0.5692802204109746, 0.001, 0.0001],
    #    [0.665939616576742, 0.0005, 0.0001],
    #    [0.8422683962805648, 0.0001, 0.0001],
    #    [0.6321891860865573, 0.001, 0.001],
    #    [0.6278268855470095, 0.0005, 0.001],
    #    [0.8263115600964298, 0.0001, 0.001],
    #]

    #_3d_scat(s32, s64, s128)
    #resnet_stag()

    #model = resnet18(num_classes=23, in_channels=1)
    #model.load_state_dict(
    #    torch.load(
    #        './checkpoints/__ResNet__RMSprop__(32,0.0001,0.0001).pt', map_location='cpu', weights_only=True
    #    )
    #)
    #plot_conf_mat(model)
    #ast_tonan()
    #ast_rms_best()
    #model = resnet18(num_classes=3, in_channels=1)
    #model = ASTModel(label_dim=3, model_size='tiny224')
    model = ASRModel(num_classes=23)
    model.load_state_dict(
        torch.load(
            #'./checkpoints/__ASTModel__Adam____3_CLASS__(32,0.0005,0.0001).pt', map_location='cpu', weights_only=True
            #'./checkpoints/__ResNet__RMSprop____3_CLASS__(128,0.0001,0)_oversampling.pt', map_location='cpu', weights_only=True
            './checkpoints/__ASRModel__Adam__(128,0.0005,0.0001).pt', map_location='cpu', weights_only=True
        )
    )
    plot_conf_mat(model)
