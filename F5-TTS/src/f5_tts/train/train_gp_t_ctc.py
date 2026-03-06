# training script.

import os
from importlib.resources import files
# import sys
# # 获取项目根目录的绝对路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# # 将项目根目录添加到 Python 的搜索路径
# sys.path.insert(0, project_root)

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM_GP_CTC, Trainer_GP_CTC
from f5_tts.model.dataset import load_dataset_gp_t
from f5_tts.model.utils import get_tokenizer_gp_t

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base_debug_LibriTTS_gp_train_ctc.yaml")
# @hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name="F5TTS_v1_Base_debug_LibriTTS_char.yaml")
def main(model_cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer_gp_t(tokenizer_path, tokenizer)

    # set model
    model = CFM_GP_CTC(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # init trainer
    trainer = Trainer_GP_CTC(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        tokenizer=tokenizer,
    )

    train_dataset = load_dataset_gp_t(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec, layer_indices_ctc=model_cfg.model.arch.layer_indices_ctc)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
