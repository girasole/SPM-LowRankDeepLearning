from torchvision.datasets import MNIST
from torch.optim import SGD, Adam, Adagrad, RMSprop
from torch import nn
from plot_utils import compute_svd_series, plot_sv_series
from torch.utils.data import DataLoader

# import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA


np.random.seed(3407)
# orthogonal linear model
# class orthinit_network(nn.Module):
#     def __init__(self, nonlinear=False, init_scale=1, hidden_dim=784, large=False, bn=False):
#         super().__init__()
#         self.nonlinear = nonlinear
#         self.d = hidden_dim
#         self.bns = None
#         self.mlps = nn.Sequential(
#             nn.Linear(784, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),

#             nn.Linear(self.d, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),
#             nn.Linear(self.d, self.d, bias=False),

#             nn.Linear(self.d, 10, bias=False),
#             )
#         if bn:
#             self.bns = nn.Sequential(
#                 None,
#                 None,
#                 None,
#                 nn.BatchNorm1d(self.d),
#                 None,
#                 None,
#                 None,
#                 nn.BatchNorm1d(self.d),
#                 None                
#             )
#         for layer in self.mlps:
#             if isinstance(layer, nn.Linear):
#                 nn.init.orthogonal_(layer.weight, gain=init_scale)
#                 # nn.init.normal_(layer.weight, mean=0, std=init_scale)
    
#     def forward(self, x):
#         for i, layer in enumerate(self.mlps):
#             # print(x.shape)
#             x = layer(x)
#             if self.nonlinear and i < len(self.mlps)-1:
#                 # if self.bns is not None:
#                 #     x = self.bns[i](x)
#                 if i in [3, 7]:
#                     if self.bns[i] is not None:
#                         x = self.bns[i](x)
#                     x = F.relu(x)
#         return x

class orthinit_network(nn.Module):
    def __init__(self, nonlinear=False, init_scale=1, hidden_dim=784, large=False, bn=False):
        super().__init__()
        self.nonlinear = nonlinear
        self.d = hidden_dim
        self.bns = None
        self.mlps = nn.Sequential(
            nn.Linear(784, self.d, bias=False),
            nn.Linear(self.d, self.d, bias=False),

            nn.Linear(self.d, self.d, bias=False),
            nn.Linear(self.d, self.d, bias=False),

            nn.Linear(self.d, 10, bias=False),
            )
        if bn:
            self.bns = nn.Sequential(
                None,
                nn.BatchNorm1d(self.d),
                None,
                nn.BatchNorm1d(self.d),
                None                
            )
        for layer in self.mlps:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=init_scale)
                # nn.init.normal_(layer.weight, mean=0, std=init_scale)
    
    def forward(self, x):
        for i, layer in enumerate(self.mlps):
            # print(x.shape)
            x = layer(x)
            if self.nonlinear and i < len(self.mlps)-1:
                # if self.bns is not None:
                #     x = self.bns[i](x)
                if i in [1, 3]:
                    if self.bns is not None:
                        x = self.bns[i](x)
                    x = F.relu(x)
        return x
    


# MNIST dataset
trainset = MNIST(root='./data', download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    torch.flatten]))
testset = MNIST(root='./data', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    torch.flatten]))

train_imgs, train_labels = [], []
test_imgs, test_labels = [], []

i=0
num_samples = 1000
idx = np.random.choice(len(trainset), num_samples, replace=False)
for img, label in tqdm(trainset):
    if i in idx:
        train_imgs.append(img.cpu().numpy())
        train_labels.append(label)
    i += 1
train_imgs = np.stack(train_imgs)
train_labels = np.stack(train_labels)


for img, label in tqdm(testset):
    test_imgs.append(img.cpu().numpy())
    test_labels.append(label)
test_imgs = np.stack(test_imgs)
test_labels = np.stack(test_labels)


# np.savez('mnist_train.npz', imgs=imgs, labels=labels)

# whiten the data
pca = PCA(whiten=True)
train_img_whitened = pca.fit_transform(train_imgs)
test_img_whitened = pca.transform(test_imgs)


# imgs_whitened_withnoise = imgs_whitened + np.random.normal(0, 1e-3, size=imgs_whitened.shape)
# print(imgs_whitened.shape)
# print(np.linalg.norm(imgs_whitened.T@imgs_whitened-num_samples*np.eye(100), ord='fro'))
# print((imgs_whitened.T@imgs_whitened-num_samples*np.eye(100) > 1e-4).sum())
# fig, ax = plt.subplots()
# ax.matshow(imgs_whitened.T@imgs_whitened, cmap='seismic')
# plt.show()

train_imgs = torch.from_numpy(train_imgs)
train_img_whitened = torch.from_numpy(train_img_whitened)
train_labels = torch.from_numpy(train_labels)
test_imgs = torch.from_numpy(test_imgs)
test_img_whitened = torch.from_numpy(test_img_whitened)
test_labels = torch.from_numpy(test_labels)


for init_scale, bn in zip([1], [True]):
    epochs = 400
    weights = []
    loss = nn.MSELoss()
    model = orthinit_network(nonlinear=True, init_scale=init_scale, hidden_dim=2000, bn=bn)
    optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0)
    # optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=0)
    # optimizer = Adagrad(model.parameters(), lr=0.01, weight_decay=0)
    # optimizer = RMSprop(model.parameters(), lr=0.01, weight_decay=0)
    balance = []
    for epoch in range(epochs):
        # W3 = model.mlps[2].weight.clone().detach()
        # W2 = model.mlps[1].weight.clone().detach()
        # W1 = model.mlps[0].weight.clone().detach()
        # print(W3.shape, W2.shape)
        # print("balance:", torch.norm((W3.T@W3 - W2@W2.T)/(torch.norm(W3.T@W3, p='fro')), p='fro'))
        # balance.append(torch.norm(W3.T@W3 - W2@W2.T, p='fro'))
        if epoch % 20 == 0: 
            weights.append(model.mlps[0].weight.clone().detach().numpy())
        losses = []
        x = train_imgs
        y = train_labels
        y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
        y_hat = model(x)
        l = loss(y_hat, y1)
        losses.append(l.item())
        pred = y_hat.argmax(dim=1)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (pred == train_labels).sum()/ train_img_whitened.shape[0]
        # if l.item() < tol:
        #     break
        print(f'Epoch {epoch+1}/{epochs} | train_Loss: {np.array(losses).mean():.4f}| train_Accuracy: {acc:.4f}')

        # pred
        with torch.no_grad():
            x = test_imgs
            y = test_labels
            y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
            y_hat = model(x)
            l = loss(y_hat, y1)
            losses.append(l.item())
            pred = y_hat.argmax(dim=1)
            acc = (pred == test_labels).sum()/ test_img_whitened.shape[0]
            print(f'            |  test_Loss: {np.array(losses).mean():.4f}| test_Accuracy: {acc:.4f}')



    # compute the difference of weights
    weights = np.stack(weights)
    series = compute_svd_series(weights, rank=10)
    sval_series, right_series, left_series = series
    sval_series = sval_series[:, :40]

    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(131, projection='3d')
    plot_sv_series(ax, sval_series)

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n$\sigma_i(t)$', fontsize=10)

    ax.set_title('Singular Values', fontsize=15, y=0.0)

    ax = fig.add_subplot(132, projection='3d')

    plot_sv_series(ax, right_series, color='inferno')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=10)

    ax.set_title('Right Singular Vectors', fontsize=15, y=0.0)

    ax = fig.add_subplot(133, projection='3d')

    plot_sv_series(ax, left_series, color='cividis')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=10)

    ax.set_title('Left Singular Vectors', fontsize=15, y=0.0)

    # plt.tight_layout()
    # plt.savefig(f'./figs/orthinit_init{init_scale}_bn{bn}_60000.png')
    plt.show()
    plt.close()


    # weights = np.stack(weights)
    weights_diff = weights[1:] - weights[:-1]


    series = compute_svd_series(weights_diff, rank=10)
    sval_series, right_series, left_series = series
    sval_series = sval_series[:, :40]

    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(131, projection='3d')
    plot_sv_series(ax, sval_series)

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n$\sigma_i(t)$', fontsize=10)

    ax.set_title('Singular Values', fontsize=15, y=0.0)

    ax = fig.add_subplot(132, projection='3d')

    plot_sv_series(ax, right_series, color='inferno')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=10)

    ax.set_title('Right Singular Vectors', fontsize=15, y=0.0)

    ax = fig.add_subplot(133, projection='3d')

    plot_sv_series(ax, left_series, color='cividis')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=10)

    ax.set_title('Left Singular Vectors', fontsize=15, y=0.0)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'./figs/orthlinit_init{init_scale}_bn{bn}_diffnn+1_60000.png')
    plt.show()
    plt.close()



    weights_diff = weights[1:] - weights[0]


    series = compute_svd_series(weights_diff, rank=10)
    sval_series, right_series, left_series = series
    sval_series = sval_series[:, :40]

    fig = plt.figure(figsize=(21, 7))
    ax = fig.add_subplot(131, projection='3d')
    plot_sv_series(ax, sval_series)

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n$\sigma_i(t)$', fontsize=10)

    ax.set_title('Singular Values', fontsize=15, y=0.0)

    ax = fig.add_subplot(132, projection='3d')

    plot_sv_series(ax, right_series, color='inferno')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(v_i(t), v_i(0))$', fontsize=10)

    ax.set_title('Right Singular Vectors', fontsize=15, y=0.0)

    ax = fig.add_subplot(133, projection='3d')

    plot_sv_series(ax, left_series, color='cividis')

    ax.set_xlabel('\nSV Index ($i$)', fontsize=10)
    ax.set_ylabel('\nIteration ($t$)', fontsize=10)
    ax.set_zlabel('\n' + r'$\angle(u_i(t), u_i(0))$', fontsize=10)

    ax.set_title('Left Singular Vectors', fontsize=15, y=0.0)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(f'./figs_adam/normalhinit_init{init_scale}_bn{bn}_diff0n_60000.png')
    plt.show()
    plt.close()

# plt.figure()
# plt.plot(balance)
# plt.show()









# train
# model = orthinit_network(nonlinear=False, init_scale=1)
# optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0)
# optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0)
# loss = nn.MSELoss()
# loss = nn.CrossEntropyLoss()



# errs, dists = [], []
# for i in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 6e-1, 7e-1, 7.5e-1, 8e-1, 8.5e-1, 9e-1]:
#     model = orthinit_network(nonlinear=False, init_scale=1, large=True)
#     optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0)
#     input = imgs_whitened + i*torch.randn(imgs_whitened.shape)
#     weights = []
#     errs.append(torch.norm(input.T@input - num_samples*torch.eye(100), p='fro'))
#     # errs.append((input.T@input - num_samples*torch.eye(100) > 1e-4).sum())
#     for epoch in range(epochs):
#         if epoch % 10 == 0: 
#             weights.append(model.mlps[0].weight.clone().detach().numpy())
#         losses = []
#         # x = imgs_whitened
#         x = input
#         y = labels
#         # print(x.shape, y.shape)
#         y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
#         y_hat = model(x)
#         # print(y_hat.shape)
#         l = loss(y_hat, y1)
#         losses.append(l.item())
#         pred = y_hat.argmax(dim=1)
#         l.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         acc = (pred == labels).sum()/ imgs_whitened.shape[0]
#         print(f'Epoch {epoch+1}/{epochs} | Loss: {np.array(losses).mean():.4f}| Accuracy: {acc:.4f}')


#     series = compute_svd_series(weights, rank=10)
#     sval_series, right_series, left_series = series
#     dists.append(right_series[-1, 50])

# errs1, dists1 = [], []
# for i in [1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1, 6e-1, 7e-1, 7.5e-1, 8e-1, 8.5e-1, 9e-1]:
#     model = orthinit_network(nonlinear=False, init_scale=1, large=False)
#     optimizer = SGD(model.parameters(), lr=0.01, weight_decay=0)
#     input = imgs_whitened + i*torch.randn(imgs_whitened.shape)
#     weights = []
#     errs1.append(torch.norm(input.T@input - num_samples*torch.eye(100), p='fro'))
#     # errs.append((input.T@input - num_samples*torch.eye(100) > 1e-4).sum())
#     for epoch in range(epochs):
#         if epoch % 10 == 0: 
#             weights.append(model.mlps[0].weight.clone().detach().numpy())
#         losses = []
#         # x = imgs_whitened
#         x = input
#         y = labels
#         # print(x.shape, y.shape)
#         y1 = nn.functional.one_hot(y, num_classes=10).to(dtype=torch.float32)
#         y_hat = model(x)
#         # print(y_hat.shape)
#         l = loss(y_hat, y1)
#         losses.append(l.item())
#         pred = y_hat.argmax(dim=1)
#         l.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         acc = (pred == labels).sum()/ imgs_whitened.shape[0]
#         print(f'Epoch {epoch+1}/{epochs} | Loss: {np.array(losses).mean():.4f}| Accuracy: {acc:.4f}')


#     series = compute_svd_series(weights, rank=10)
#     sval_series, right_series, left_series = series
#     dists1.append(right_series[-1, 50])

# plt.figure()
# plt.plot(errs, dists, 'o-', color='r', label='small'), plt.plot(errs1, dists1, 'o-', color='b', label='large')
# plt.xlabel('XX^T-I'), plt.ylabel('subspace shift')
# plt.legend(), plt.show()