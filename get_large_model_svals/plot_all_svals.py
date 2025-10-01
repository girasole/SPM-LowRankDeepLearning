import matplotlib.pyplot as plt
from plot_utils import plot_sv_series, plot_sv_series_2d
import numpy as np
from numpy.linalg import svd

weights_linear = np.load('linear_orth.npz')['weights'] # (epochs, sval_num)
weights_MLP = np.load('MLP.npz')['weights'] # (epochs, sval_num)
weights_ViT = np.load('ViT.npz')['weights'] # (epochs, sval_num)
weights_VGG = np.load('VGG16.npz')['weights'] # (epochs, sval_num)

print(weights_linear.shape)
print(weights_MLP.shape)
print(weights_VGG.shape)
print(weights_ViT.shape)

def compute_svd_series(weights):
    sval_series = []
    for w in  weights[0:]:
        _, st, _ = svd(w, full_matrices=False)
        st = st/np.max(st)
        print(st.shape)
        sval_series.append(st)
    sval_series = np.array(sval_series)
    
    return sval_series


# svals_linear = [compute_svd_series(weights_linear), 
#                 compute_svd_series(weights_linear[1:]-weights_linear[0])]
# svals_MLP = [compute_svd_series(weights_MLP),
#                 compute_svd_series(weights_MLP[1:]-weights_MLP[0])]
# svals_toytransformer = [compute_svd_series(weights_toytransformer),
#                 compute_svd_series(weights_toytransformer[1:]-weights_toytransformer[0])]
# svals_ViT = [compute_svd_series(weights_ViT),
#                 compute_svd_series(weights_ViT[1:]-weights_ViT[0])]

svals_linear = compute_svd_series(weights_linear[1:]-weights_linear[0])
svals_MLP = compute_svd_series(weights_MLP[1:]-weights_MLP[0])
svals_toytransformer = compute_svd_series(weights_VGG[1:]-weights_VGG[0])
svals_ViT = compute_svd_series(weights_ViT[1:]-weights_ViT[0])


# subfigs_name = ['W(t)-W(0)']
# subfigs_color = ['viridis']


# fig = plt.figure(constrained_layout=True, figsize=(28, 8))
# subfigs = fig.subfigures(nrows=1, ncols=1)
# for i, subfig in enumerate(subfigs):
#     subfig.suptitle(subfigs_name[i], fontsize=40, weight='bold', y=0.92)


#     ax = subfig.add_subplot(141, projection='3d')
#     plot_sv_series(ax, svals_linear[i][:, :50], color=subfigs_color[i])
#     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)
#     # if i == 0:
#     #     ax.set_zlim(0.85, 1.05)
#     ax.set_title('Linear Model', fontsize=25, y=0.0)


#     ax = subfig.add_subplot(142, projection='3d')
#     plot_sv_series(ax, svals_MLP[i][:, :50], color=subfigs_color[i])
#     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)
#     ax.set_title('MLP', fontsize=25, y=0.0)

#     ax = subfig.add_subplot(143, projection='3d')
#     plot_sv_series(ax, svals_toytransformer[i][:, :50], color=subfigs_color[i])
#     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)
#     ax.set_title('Toy ViT', fontsize=25, y=0.0)
    

#     ax = subfig.add_subplot(144, projection='3d')
#     plot_sv_series(ax, svals_ViT[i][:20], sample_step=15, color=subfigs_color[i])
#     ax.set_xlabel('\nSV Index ($i$)', fontsize=20)
#     ax.set_ylabel('\nIteration ($t$)', fontsize=20)
#     ax.set_zlabel('\n$\sigma_i(t)$', fontsize=20)
#     ax.set_title('ViT-B', fontsize=25, y=0.0)
    


# plt.tight_layout()
# plt.savefig('./proposal/svals_all_1.png')
# plt.close()


subfigs_name = 'W(t)-W(0)'
subfigs_color = 'viridis'


fig = plt.figure(constrained_layout=True, figsize=(27, 5.5))

# --- 1st subplot ---
ax0 = fig.add_subplot(141)
plot_sv_series_2d(ax0, svals_linear[:10, :40], color=subfigs_color)
ax0.set_xlabel('Index', fontsize=20)
ax0.set_ylabel('$\\sigma_i(t)$', fontsize=20)
ax0.tick_params(axis='both', which='major', labelsize=20)
ax0.set_title('Linear Model', fontsize=30, y=-0.3)

# capture legend entries from the first plot
handles, labels = ax0.get_legend_handles_labels()

# --- 2nd subplot ---
ax1 = fig.add_subplot(142)
plot_sv_series_2d(ax1, svals_MLP[:10, :40], color=subfigs_color)
ax1.set_xlabel('Index', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_title('MLP', fontsize=30, y=-0.3)
# no legend call here

# --- 3rd subplot ---
ax2 = fig.add_subplot(143)
plot_sv_series_2d(ax2, svals_toytransformer[:10, :40], color=subfigs_color)
ax2.set_xlabel('Index', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.set_title('VGG', fontsize=30, y=-0.3)
# no legend call here

# --- 4th (rightmost) subplot ---
ax3 = fig.add_subplot(144)
plot_sv_series_2d(ax3, svals_ViT[:10, :40], color=subfigs_color)
ax3.set_xlabel('Index', fontsize=20)
ax3.tick_params(axis='both', which='major', labelsize=20)
ax3.set_title('ViT-B', fontsize=30, y=-0.3)

# ONLY put the legend on the rightmost plot, using labels from the first
ax3.legend(handles, labels, ncol=2, fontsize=15)

plt.tight_layout()
plt.savefig('.svals_all.pdf', bbox_inches='tight', dpi=500)
plt.close()

