from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn.functional as F
import torchaudio
from datetime import timedelta
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from f5_tts.infer.utils_infer import load_vocoder

from f5_tts.model.dataset import DynamicBatchSampler, collate_fn_eval_metric
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

class Evaluator_metric:
    def __init__(
        self,
        metric: str = "utmos",
        batch_size_per_gpu: int = 38400,
        batch_size_type: str = "frame",
        max_samples: int = 64,
        accelerate_kwargs: dict = dict(),
        vocoder_local_path: str = "",
        wavlm_ckpt_path: str = "",
        vocoder_sample_rate: int = 24000,
        skip = True,
    ):
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=172800))
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, process_group_kwargs],
            **accelerate_kwargs,
        )
        
        self.metric = metric
        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.vocoder_sample_rate = vocoder_sample_rate
        
        # 加载 Vocoder
        self.vocoder = load_vocoder(
            vocoder_name="vocos",
            is_local=True,
            local_path=vocoder_local_path,
            device=self.accelerator.device
        )
        
        # Resampler: 24k -> 16k
        self.resampler = torchaudio.transforms.Resample(vocoder_sample_rate, 16000).to(self.accelerator.device)
        self.skip = skip
        
        # 加载评估模型
        if metric == "utmos":
            self._load_utmos()
        elif metric == "sim":
            self._load_sim(wavlm_ckpt_path)

    def _load_utmos(self):
        os.environ['TORCH_HUB_DISABLE_VALIDATION'] = '1'
        torch.hub.set_dir('/.cache/torch/hub')
        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True, skip_validation=True)
        self.model = self.model.to(self.accelerator.device).eval()

    def _load_sim(self, ckpt_path):
        self.model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model = self.model.to(self.accelerator.device).eval()

    def evaluate(self, dataset, num_workers=16, resumable_with_seed=666):
        # DataLoader 构建逻辑
        if self.batch_size_type == "sample":
            dataloader = DataLoader(dataset, collate_fn=collate_fn_eval_metric, batch_size=self.batch_size_per_gpu, num_workers=num_workers, shuffle=False)
        else:
            self.accelerator.even_batches = False
            sampler = SequentialSampler(dataset)
            batch_sampler = DynamicBatchSampler(sampler, self.batch_size_per_gpu, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_residual=False, drop_last=False)
            dataloader = DataLoader(dataset, collate_fn=collate_fn_eval_metric, batch_sampler=batch_sampler, num_workers=num_workers)
        
        dataloader = self.accelerator.prepare(dataloader)
        
        for batch in tqdm(dataloader, desc=f"Evaluating {self.metric}", disable=not self.accelerator.is_local_main_process):
            self.process_batch(batch)
        
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(f"Finished evaluating {self.metric.upper()}!")

    def process_batch(self, batch):
        gen_mels = batch["gen_mel"] # (B, Mel, T_padded)
        mel_lengths = batch["mel_lengths"]
        stems = batch["mel_path_stems"]
        suffix = f"_{self.metric}.txt"

        # 检查是否全部已处理，跳过以节省时间
        if all(os.path.exists(s + suffix) for s in stems) and self.skip:
            return

        with torch.inference_mode():
            # 1. 批量解码 (高效!)
            # Vocos 输入需要 float32
            # gen_wavs_padded = self.vocoder.decode(gen_mels.float()) # (B, 1, T_wav_padded)
            gen_mels = gen_mels.to(self.accelerator.device).float()
            gen_wavs_padded = self.vocoder.decode(gen_mels)
            # 获取当前 Batch 生成的实际最大长度
            actual_max_len = gen_wavs_padded.shape[-1]
            
            # 2. 逐个切片并评分 (准确!)
            for i, stem in enumerate(stems):
                save_path = stem + suffix
                if os.path.exists(save_path) and self.skip: continue

                # 切片：去掉 Padding
                theoretical_len = mel_lengths[i].item() * 256
                # 实际有效长度：取理论值和实际生成值的最小值
                valid_samples = min(theoretical_len, actual_max_len)
                wav_valid = gen_wavs_padded[i, :valid_samples].unsqueeze(0) # (1, T_valid)
                # ========== 新增：保存 wav 文件用于测试 ==========
                # wav_save_dir = "/vepfs/group09/F5-TTS_qingyu_MultilingualTTS/tmp"
                # os.makedirs(wav_save_dir, exist_ok=True)
                # wav_filename = os.path.basename(stem) + ".wav"
                # wav_save_path = os.path.join(wav_save_dir, wav_filename)
                # torchaudio.save(wav_save_path, wav_valid.cpu(), self.vocoder_sample_rate)
                # ================================================

                score = 0.0
                try:
                    if self.metric == "utmos":
                        score = self.model(wav_valid, self.vocoder_sample_rate).item()
                    
                    elif self.metric == "sim":
                        # 重采样到 16k
                        wav_16k = self.resampler(wav_valid)

                        # 获取对应的参考音频
                        ref_wav = batch["ref_audio"][i].to(self.accelerator.device) # (T_ref_padded)
                        ref_len = batch["ref_lengths"][i].item()
                        ref_wav = ref_wav[:ref_len].unsqueeze(0) # (1, T_valid_ref)

                        # 提取 Embedding
                        emb_gen = self.model(wav_16k)
                        emb_ref = self.model(ref_wav) # ref 已经在 Dataset 里重采样过 16k 了
                        score = F.cosine_similarity(emb_gen, emb_ref)[0].item()
                except (RuntimeError, ValueError) as e:
                    error_msg = str(e)
                    if "Kernel size can't be greater than actual input size" in error_msg or "Expected more than 1 spatial element" in error_msg:
                    # 捕获音频太短导致的卷积错误
                        print(f"Skipping {stem}: audio too short ({valid_samples} samples)")
                        score = -1.0
                    else:
                        raise e

                # 写入
                with open(save_path, "w") as f:
                    f.write(f"{score:.6f}")