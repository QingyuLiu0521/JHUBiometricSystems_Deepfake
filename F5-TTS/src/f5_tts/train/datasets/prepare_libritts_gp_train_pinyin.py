import os
import sys
import torch

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
from f5_tts.model.utils import convert_char_to_pinyin

def deal_with_audio_dir(audio_dir, mel_dir):
    sub_result, durations = [], []
    prompt_frames_list = []
    vocab_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue

        audio_dir_suffix = '/'.join(line.parts[-4:-1])
        audio_filename = line.stem
        rel_path = f"{audio_dir_suffix}/{audio_filename}"
        
        json_path = os.path.join(mel_dir, f"{rel_path}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            json_obj = json.load(f)
        prompt_frames = json_obj["gen_len"]
        gen_text = json_obj["text"]
        total_text = gen_text + " " + text
        
        if tokenizer == "pinyin":
            total_text = convert_char_to_pinyin([total_text], polyphone=polyphone)[0]
            text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
        pt_path = os.path.join(mel_dir, f"{rel_path}.pt")
        if not os.path.exists(pt_path):
            continue

        sub_result.append({"audio_path": str(line), "text": text, "total_text": total_text, "duration": duration, "mel_path": pt_path, "prompt_frames": prompt_frames})
        durations.append(duration)
        prompt_frames_list.append(prompt_frames)
        vocab_set.update(list(text))
    return sub_result, durations, prompt_frames_list, vocab_set


def main():
    result = []
    duration_list = []
    prompt_frames_list = []
    text_vocab_set = set()

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir, mel_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, prompt_frames, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        prompt_frames_list.extend(prompt_frames)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list, "prompt_frames": prompt_frames_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    base_vocab_path = str(files("f5_tts").joinpath("../../")) + f"/data/Emilia_ZH_EN_pinyin_gp_t/vocab.txt"
    if os.path.exists(base_vocab_path) and tokenizer == "pinyin":
        import shutil
        shutil.copy(base_vocab_path, f"{save_dir}/vocab.txt")
        print(f"Copied vocab.txt from {base_vocab_path}")
    else:
        with open(f"{save_dir}/vocab.txt", "w") as f:
            for vocab in sorted(text_vocab_set):
                f.write(vocab + "\n")
        print(f"Created new vocab.txt")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "pinyin"  # "pinyin" | "char"
    polyphone = True

    SUB_SET = ["train-clean-100"]
    dataset_dir = "/hpc_stor03/sjtu_home/qingyu.liu/LibriTTS"
    mel_dir = "/hpc_stor03/sjtu_home/qingyu.liu/F5-TTS-main/LibriTTS_100_gen"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}_gp_t"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
