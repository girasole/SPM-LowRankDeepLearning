from torchvision.datasets import MNIST
from torch.optim import SGD, Adam, Adagrad, RMSprop
from torch import nn
from utils import compute_svd_series, plot_sv_series, compute_angle
from torch.utils.data import DataLoader

# import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
    
np.random.seed(3407)
class orthinit_network(nn.Module):
    def __init__(self, nonlinear=False, init_scale=1, hidden_dim=784, large=False, bn=False, init_type=None):
        super().__init__()
        self.nonlinear = nonlinear
        self.d = hidden_dim
        self.bns = None
        self.mlps = nn.Sequential(
            nn.Linear(784, self.d, bias=False),
            nn.Linear(self.d, self.d, bias=False),
            nn.Linear(self.d, self.d, bias=False),
            )
        self.out_fc = nn.Linear(self.d, 10, bias=False)
        if bn:
            self.bns = nn.Sequential(
            nn.BatchNorm1d(self.d, affine=False),
            nn.BatchNorm1d(self.d, affine=False),
            nn.BatchNorm1d(self.d, affine=False),
            )
        # self.out_bn = nn.BatchNorm1d(10, affine=False)
        # self.out_norm = nn.LayerNorm(10, elementwise_affine=False)
        # self.mlps[-1].weight.requires_grad = False
        if init_type == 'orth':
            for layer in self.mlps:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=init_scale)
        elif init_type == 'normal':
            for layer in self.mlps:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=init_scale)
                    # nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        elif init_type == 'uniform':
            for layer in self.mlps:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, -init_scale, init_scale)
        elif init_type == 'eye':
            for layer in self.mlps:
                if isinstance(layer, nn.Linear):
                    nn.init.eye_(layer.weight)
    
    def forward(self, x):
        for i, layer in enumerate(self.mlps):
            # print(x.shape)
            x = layer(x)
            # if i < len(self.mlps)-2:
            if self.bns is not None:
                x = self.bns[i](x)
            if self.nonlinear:
                x = F.relu(x)
            # if i == len(self.mlps)-1:
            #     # x = self.out_bn(x)
            #     x = self.out_norm(x)
        x = self.out_fc(x)
        return x
    

# MNIST dataset
trainset = MNIST(root='./data', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    torch.flatten]))
testset = MNIST(root='./data', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    torch.flatten]))

trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
testloader = DataLoader(testset, batch_size=100, shuffle=False)


figs_dir = './AISTATS/'
# figs_dir = './figs_batchgrad_812/'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)


for init_scale, bn, type, nonlinear, lr in zip([1e-1], [False], ['orth'], [False], [1e-1]):
    epochs = 40
    weights_all = [[], [], []]
    weight_penult = []
    loss = nn.MSELoss()
    model = orthinit_network(nonlinear=nonlinear, init_scale=init_scale, hidden_dim=784, bn=bn, init_type=type)
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=0)
    for epoch in range(epochs):
        if epoch % 2 == 0:
            for i in [0, 1, 2]: 
                weights_all[i].append(model.mlps[i].weight.clone().detach().numpy())
            weight_penult.append(model.mlps[-2].weight.clone().detach().numpy())
        losses, labels, preds = [], [], []
        model.train()


        # cls_head = model.mlps[-1].weight.clone().detach().numpy()
        # # cls_head_0 = model.mlps[-1].weight.clone().detach().numpy()
        # Wl = model.mlps[-2].weight.clone().detach().numpy()
        # # V = np.zeros((2000, 10))
        # _, _, V = np.linalg.svd(cls_head, full_matrices=True)
        # V = V.T

        for x, y in tqdm(trainloader):
            y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
            y_hat = model(x)
            pred = y_hat.argmax(dim=1)
            l = loss(y_hat, y1)
            losses.append(l.item())
            preds.append(pred.cpu().numpy())
            labels.append(y.cpu().numpy())

            # with torch.no_grad():
            #     cls_head_new = model.mlps[-1].weight.clone().detach().numpy()
            #     # Wl_new = model.mlps[-2].weight.clone().detach().numpy()
            #     # U_new, _, _ = np.linalg.svd(Wl_new-Wl)
            #     # U_new = U_new[:, :10]
            #     _, _, V_new = np.linalg.svd(cls_head_new)
            #     V_new = V_new.T[:, :10]
            #     print(compute_angle(V_new, V))
            #     # print("cls_head shift:", compute_angle(cls_head_new.T, cls_head.T)) 
            #     # print(compute_angle(U[:, :10], cls_head.T))
            #     V = V_new
            #     # Wl_new = Wl
            #     cls_head = cls_head_new


            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc = (np.stack(preds, axis=0) == np.stack(labels, axis=0)).sum()/ len(trainset)       
        print(f'Epoch {epoch+1}/{epochs} | train_Loss: {np.array(losses).mean():.4f}| train_Accuracy: {acc:.4f}')

        # pred
        with torch.no_grad():
            losses, labels, preds = [], [], []
            model.eval()
            for x, y in tqdm(testloader):
                y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
                y_hat = model(x)
                y_hat = model(x)
                pred = y_hat.argmax(dim=1)
                l = loss(y_hat, y1)
                losses.append(l.item())
                preds.append(pred.cpu().numpy())
                labels.append(y.cpu().numpy())

            acc = (np.stack(preds, axis=0) == np.stack(labels, axis=0)).sum()/ len(testset)
            print(f'Epoch {epoch+1}/{epochs} | test_Loss: {np.array(losses).mean():.4f}| test_Accuracy: {acc:.4f}')



    weights_penult = np.stack(weight_penult, axis=0)
    np.savez('weight_linear.npz', weights=weights_penult)
    exit()
    print('weight:')
    series_list = []
    for i in [0, 1, 2]:
        series_list.append(compute_svd_series(weights_all[i], rank=10))
    
    sample_step = 15
    sigval_num = 40


    fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for i, subfig in enumerate(subfigs):

        sval_series, right_series, left_series = series_list[i]
        subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
        ax = subfig.add_subplot(131, projection='3d')
        plot_sv_series(ax, sval_series, sample_step=sample_step)

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

        ax.set_title('Singular Values', fontsize=25, y=0.0)

        ax = subfig.add_subplot(132, projection='3d')

        plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

        ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

        ax = subfig.add_subplot(133, projection='3d')

        plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

        ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    plt.tight_layout()
    plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_subsampled{sample_step}.png')
    plt.close()


    fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    subfigs = fig.subfigures(nrows=3, ncols=1)
    for i, subfig in enumerate(subfigs):

        sval_series, right_series, left_series = series_list[i]
        sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
        subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
        ax = subfig.add_subplot(131, projection='3d')
        plot_sv_series(ax, sval_series)

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

        ax.set_title('Singular Values', fontsize=25, y=0.0)

        ax = subfig.add_subplot(132, projection='3d')

        plot_sv_series(ax, right_series, color='inferno')

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

        ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

        ax = subfig.add_subplot(133, projection='3d')

        plot_sv_series(ax, left_series, color='cividis')

        ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
        ax.set_ylabel('\nIteration ($t$)', fontsize=20)
        ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

        ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    plt.tight_layout()
    plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_first{sigval_num}.png')
    plt.close()
    

    # print('weight diffnn+1:')
    # series_list = []
    # for i in [0, 1, 2]:
    #     weight = weights_all[i]
    #     weight = np.stack(weight, axis=0)
    #     weight = weight[1:] - weight[:-1]
    #     series_list.append(compute_svd_series(weight, rank=10))
    
    # fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    # subfigs = fig.subfigures(nrows=3, ncols=1)
    # for i, subfig in enumerate(subfigs):

    #     sval_series, right_series, left_series = series_list[i]
    #     subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
    #     ax = subfig.add_subplot(131, projection='3d')
    #     plot_sv_series(ax, sval_series, sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

    #     ax.set_title('Singular Values', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(132, projection='3d')

    #     plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

    #     ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(133, projection='3d')

    #     plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

    #     ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    # plt.tight_layout()
    # plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_diffnn+1_subsampled{sample_step}.png')
    # plt.close()
    
    # fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    # subfigs = fig.subfigures(nrows=3, ncols=1)
    # for i, subfig in enumerate(subfigs):

    #     sval_series, right_series, left_series = series_list[i]
    #     sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
    #     subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
    #     ax = subfig.add_subplot(131, projection='3d')
    #     plot_sv_series(ax, sval_series)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

    #     ax.set_title('Singular Values', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(132, projection='3d')

    #     plot_sv_series(ax, right_series, color='inferno')

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

    #     ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(133, projection='3d')

    #     plot_sv_series(ax, left_series, color='cividis')

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

    #     ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    # plt.tight_layout()
    # plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_diffnn+1_first{sigval_num}.png')
    # plt.close()


    # print('weight diff0n:')
    # series_list = []
    # for i in [0, 1, 2]:
    #     weight = weights_all[i]
    #     weight = np.stack(weight, axis=0)
    #     weight = weight[1:] - weight[0]
    #     series_list.append(compute_svd_series(weight, rank=10))
    
    # fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    # subfigs = fig.subfigures(nrows=3, ncols=1)
    # for i, subfig in enumerate(subfigs):

    #     sval_series, right_series, left_series = series_list[i]
    #     subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
    #     ax = subfig.add_subplot(131, projection='3d')
    #     plot_sv_series(ax, sval_series, sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

    #     ax.set_title('Singular Values', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(132, projection='3d')

    #     plot_sv_series(ax, right_series, color='inferno', sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

    #     ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(133, projection='3d')

    #     plot_sv_series(ax, left_series, color='cividis', sample_step=sample_step)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

    #     ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    # plt.tight_layout()
    # plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_diff0n_subsampled{sample_step}.png')
    # plt.close()

    # fig = plt.figure(constrained_layout=True, figsize=(21, 24))
    # subfigs = fig.subfigures(nrows=3, ncols=1)
    # for i, subfig in enumerate(subfigs):

    #     sval_series, right_series, left_series = series_list[i]
    #     sval_series, right_series, left_series = sval_series[:, :sigval_num], right_series[:, :sigval_num], left_series[:, :sigval_num]
    #     subfig.suptitle(f'Layer {i+1}', fontsize=40, weight='bold', y=0.92)
    #     ax = subfig.add_subplot(131, projection='3d')
    #     plot_sv_series(ax, sval_series)

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)

    #     ax.set_title('Singular Values', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(132, projection='3d')

    #     plot_sv_series(ax, right_series, color='inferno')

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=20)

    #     ax.set_title('Right Singular Vectors', fontsize=25, y=0.0)

    #     ax = subfig.add_subplot(133, projection='3d')

    #     plot_sv_series(ax, left_series, color='cividis')

    #     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
    #     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
    #     ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=20)

    #     ax.set_title('Left Singular Vectors', fontsize=25, y=0.0)

    # plt.tight_layout()
    # plt.savefig(f'./{figs_dir}/nonlinear{nonlinear}_{type}_{init_scale}_bn{bn}_lr{lr}_diff0n_first{sigval_num}.png')
    # plt.close()