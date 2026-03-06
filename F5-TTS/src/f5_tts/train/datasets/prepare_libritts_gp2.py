import os
import sys
from datasets import Dataset
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 将项目根目录添加到 Python 的搜索路径
sys.path.insert(0, project_root)

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue

        # 构造rel_path: audio_dir的最后两部分 / line（去掉.wav）
        audio_dir_suffix = '/'.join(line.parts[-4:-1])
        audio_filename = line.stem  # 去掉.wav后缀的文件名
        rel_path = f"{audio_dir_suffix}/{audio_filename}" # 例'train-clean-100/19/198/19_198_000000_000000.wav'

        sub_result.append({
            "audio_path": str(line), 
            "text": text, 
            "duration": duration,
            "rel_path": rel_path
        })
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    print(f"\nSaving to {save_dir} ...")
    
    # 直接将内存中的 result 列表转换为 Dataset 对象
    ds = Dataset.from_list(result)
    
    # 存盘 (save_to_disk 会自动生成 arrow 文件和 dataset_info.json)
    # 这里的 save_dir 是整个目录，它会在里面生成 data-00000-of-xxxxx.arrow 等文件
    # 为了兼容你现在的读取逻辑 (dataset.py 读取 raw 文件夹或 raw.arrow)，
    # 我们直接把 Dataset 保存到 save_dir/raw 目录下
    
    raw_save_path = os.path.join(save_dir, "raw")
    ds.save_to_disk(raw_save_path)
    
    print(f"✅ Saved {len(ds)} samples to {raw_save_path}")

    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100"]
    # SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = "/hpc_stor03/sjtu_home/qingyu.liu/LibriTTS"
    # dataset_dir = "<SOME_PATH>/LibriTTS"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = "/hpc_stor03/sjtu_home/qingyu.liu/F5-TTS-main" + f"/data/{dataset_name}_gp"
    # save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
