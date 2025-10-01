from parsimony_utils import compute_svd_series, plot_sv_series
import matplotlib.pyplot as plt
import numpy as np
import os


weights_dict = np.load('vit_timm_patch4_qkv.npz', allow_pickle=True) # change directory for stored weights
num_layers = 11
weight_size = 768
num_epochs = 10
k = 40 # Num sval indices

figs_dir_all = '.' # change directory for saving
if not os.path.exists(figs_dir_all):
    os.makedirs(figs_dir_all)

weights_all = []


for name, weight in weights_dict.items():
    weights_all.append(weight[:num_epochs, weight_size*2:, :]) # use this for qkv
    # weights_all.append(weight) # use this for fc
plot_list = [weights_all[4], weights_all[9], weights_all[10]] # plot 3 different layers

import numpy as np
import matplotlib.pyplot as plt

def plot_4panels_svals(arr_list, k=40, num_plot=10, layer_titles=None):
    """
    arr_list: list of 4 arrays, each shape (num_epochs, n, n)
    k:        number of singular values to plot
    num_plot: number of epochs to plot (first num_plot epochs)
              labeled as 2, 4, 6, ...
    layer_titles: list of 4 strings shown at the *bottom* of each subplot
                  default: ["Layer 1","Layer 2","Layer 3","Layer 4"]
    """
    assert len(arr_list) == 3, "Provide exactly 4 arrays"
    if layer_titles is None:
        layer_titles = [f"Layer {i}" for i in range(1, 5)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # consistent colormap across all panels
    max_label = 2 * num_plot
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=2, vmax=max_label)

    for i, (ax, arr, bottom_title) in enumerate(zip(axes, arr_list, layer_titles)):
        T, n, _ = arr.shape
        k_ = min(k, n)

        # compute singular values
        svals = np.empty((T, k_), dtype=float)
        for t in range(T):
            s = np.linalg.svd(arr[t], compute_uv=False, hermitian=False)
            svals[t] = s[:k_]

        # normalize by max within this subplot
        svals /= svals.max()

        # pick first num_plot epochs, but label as 2,4,6,...
        plot_indices = np.arange(min(num_plot, T))
        plot_labels = 2 * (plot_indices + 1)
        x = np.arange(1, k_ + 1)

        for idx, t in enumerate(plot_indices):
            label_val = plot_labels[idx]
            ax.plot(x, svals[t], lw=2, color=cmap(norm(label_val)), label=f"Epoch {label_val}")

        ax.set_yscale('log')
        ax.set_xlabel("Index", fontsize=18)
        if i == 0:
            ax.set_ylabel(r"$\sigma_i(t)$", fontsize=18)
        ax.set_title(bottom_title, y=-0.28, fontsize=20)
        # ax.grid(True, which="both", ls=":", alpha=0.4)
        if i == len(axes) - 1:
            ax.legend(loc="best", framealpha=0.9, fontsize=13, ncol=2)
        ax.tick_params(axis='both', labelsize=18)
    fig.tight_layout()
    plt.savefig('vit_value_2d_3plots.pdf', bbox_inches='tight', dpi=500)


plot_4panels_svals(plot_list, k=40, layer_titles=['Layer 5', 'Layer 10', 'Layer 11'])
    
