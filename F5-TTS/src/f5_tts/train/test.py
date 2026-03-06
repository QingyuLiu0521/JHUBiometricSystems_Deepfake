import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM_CTC, Trainer_CTC
from f5_tts.model.backbones.dit_ctc_ca import DiT_CTC_CA
import os
from importlib.resources import files
CA_list = ["DiT_CTC_CA"]

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base_debug_Emilia_bpe_ctc_ca.yaml")

# --- 使用示例 ---
# 实例化你的模型 (根据你的配置调整)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    model = model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels)
    def count_parameters(model):
        # 1. 计算总参数量

        total_params = sum(p.numel() for p in model.parameters())
        
        # 2. 计算可训练参数量 (requires_grad=True)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 3. 转换为 '百万' (M) 单位
        print(f"Total Parameters: {total_params / 1_000_000:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1_000_000:.2f} M")
        
        return total_params, trainable_params
    count_parameters(model)

