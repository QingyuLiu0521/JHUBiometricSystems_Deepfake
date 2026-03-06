import os
import torch
pt_path = "/hpc_stor03/sjtu_home/qingyu.liu/F5-TTS-main/LibriTTS_100_gen/train-clean-100/27/124992/27_124992_000044_000001.pt"
mel_tensor = torch.load(pt_path, map_location='cpu')
prompt_frames = mel_tensor.shape[0]
print(prompt_frames)