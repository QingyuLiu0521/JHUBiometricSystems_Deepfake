import os
import sys
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 将项目根目录添加到 Python 的搜索路径
sys.path.insert(0, project_root)

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
from datasets.arrow_writer import ArrowWriter


def deal_with_audio_dir(audio_dir):
    """处理音频目录的函数"""
    sub_result, durations = [], []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav")) # 递归获取所有wav文件

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt") # 获取对应的文本文件路径
        text = open(text_path, "r").read().strip()      # 读取文本内容
        duration = sf.info(line).duration               # 获取音频时长
        if duration < 0.4 or duration > 30:             # 过滤掉过短(<0.4)或过长(>30)的音频
            continue

        # 构造rel_path: audio_dir的最后两部分 / line（去掉.wav）
        audio_dir_suffix = '/'.join(line.parts[-4:-1])
        audio_filename = line.stem  # 去掉.wav后缀的文件名
        rel_path = f"{audio_dir_suffix}/{audio_filename}" # 例'train-clean-100/19/198/19_198_000000_000000.wav'

        # 添加数据到结果列表
        sub_result.append({
            "audio_path": str(line), 
            "text": text, 
            "duration": duration,
            "rel_path": rel_path
        })
        durations.append(duration)
        vocab_set.update(list(text))    # 更新词汇集合
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()

    # 使用多进程处理数据 process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    # 遍历数据集子集
    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        # 为每个音频目录创建处理任务
        # 列表推导式语法，等价于for循环
        [
            # 提交一个处理任务到进程池
            # 1. 第一个参数是要执行的函数（这里是deal_with_audio_dir）
            # 2. 第二个参数是传给该函数的参数（这里是audio_dir）
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    # 收集处理结果
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    # 保存为Arrow格式
    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # 保存时长信息到JSON文件 dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # 保存词汇表 vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100"]
    # SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = "/hpc_stor03/sjtu_home/qingyu.liu/LibriTTS"
    # dataset_dir = "<SOME_PATH>/LibriTTS"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = "/hpc_stor03/sjtu_home/qingyu.liu/F5-TTS-main_v104" + f"/data/{dataset_name}_gp"
    # save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
