import torch
import timm
import torch.nn as nn
import numpy as np
import os

net = timm.create_model("vit_base_patch16_384", pretrained=False)
net.head = nn.Linear(net.head.in_features, 10)

checkpoint = torch.load("log_new/new/vit_timm-4-ckpt-epoch99.pth", map_location="cpu")

# Check what keys exist in the checkpoint
print(checkpoint.keys())

net.load_state_dict(checkpoint['model'])

qkv_dict = {}


for name, param in net.named_parameters():
    print(name, param.shape)
    if "weight" in name and param.shape == torch.Size([3*768, 768]):
        qkv_dict[name]=[]
    if "weight" in name and param.shape == torch.Size([3*768, 768]):
        qkv_dict[name].append(param.detach().clone().cpu().numpy())

for name, weights in qkv_dict.items():
    qkv_dict[name] = np.stack(weights, axis=0)

np.savez(os.path.join("log_new/new", f'vit_timm_patch4_qkv'), **qkv_dict)