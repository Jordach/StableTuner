from safetensors.torch import load_file
import torch

file_path = "/hdd/zd_base0-9/vae/diffusion_pytorch_model.safetensors"
loaded = load_file(file_path)

print(loaded)