import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.eval.evaluator_metric import Evaluator_metric
from f5_tts.model.dataset import load_dataset_eval_metric

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project

abs_path = str(files("f5_tts").joinpath("../../"))


@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(eval_cfg):
    dataset_name = eval_cfg.datasets.name
    tokenizer = eval_cfg.datasets.tokenizer
    metric = eval_cfg.eval.metric
    
    # 创建评估器
    evaluator = Evaluator_metric(
        metric=metric,
        batch_size_per_gpu=eval_cfg.datasets.batch_size_per_gpu,
        batch_size_type=eval_cfg.datasets.batch_size_type,
        max_samples=eval_cfg.datasets.max_samples,
        vocoder_local_path=eval_cfg.eval.vocoder_local_path,
        wavlm_ckpt_path=eval_cfg.eval.get("wavlm_ckpt_path", None),
        vocoder_sample_rate=eval_cfg.eval.target_sample_rate,
    )
    
    # 加载数据集
    eval_dataset = load_dataset_eval_metric(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        target_sample_rate=eval_cfg.eval.target_sample_rate,
        metric=metric
    )
    
    # 开始评估
    evaluator.evaluate(
        eval_dataset,
        num_workers=eval_cfg.datasets.num_workers,
        resumable_with_seed=666,
    )


if __name__ == "__main__":
    main()