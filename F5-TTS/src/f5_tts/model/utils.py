# ruff: noqa: F722 F821

from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import json
import jieba
from pypinyin import lazy_pinyin, Style
import math
import re
from typing import List

available_texts = [
    "The quick brown fox jumps.",
    "Life is what happens to you.",
    "Knowledge is power and wisdom is strength.",
    "Time flies when you are having fun.",
    "Practice makes perfect in everything.",
    "Where there is a will there is way.",
    "Actions speak louder than words do.",
    "The early bird catches the worm.",
    "Never give up on your dreams.",
    "Learning is a lifelong journey for everyone.",
    "Success comes to those who work hard.",
    "Every cloud has a silver lining.",
    "Rome was not built in a day.",
    "A journey begins with a single step.",
    "Believe in yourself and anything is possible.",
    "The best time to plant a tree.",
    "Happiness is a choice we make daily.",
    "Great minds think alike sometimes.",
    "Tomorrow is another day to try.",
    "Stay hungry and stay foolish always.",
]

# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)

def mask_from_prompt_lens(prompt_lens: int["b"], lens: int["b"]):
    return mask_from_start_end_indices(lens, prompt_lens, lens)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]]| list[list[int]],
    vocab_char_map: dict[str, int]| None,  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def bpe_padded(
    token_ids: list[list[int]],
    padding_value: int = -1,
) -> int["b nt"]:
    list_idx_tensors = [torch.tensor(ids) for ids in token_ids]
    padded = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return padded

# Get tokenizer


def get_tokenizer(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256
    
    elif tokenizer == "bpe":
        vocab_char_map = None
        tokenizer_info_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/tokenizer_info.json")
        with open(tokenizer_info_path, "r", encoding="utf-8") as f:
            tokenizer_info = json.load(f)
        vocab_size = tokenizer_info["vocab_size"]

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

def get_tokenizer_gp_t(dataset_name, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}_gp_t/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256
    
    elif tokenizer == "bpe":
        vocab_char_map = None
        tokenizer_info_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}_gp_t/tokenizer_info.json")
        with open(tokenizer_info_path, "r", encoding="utf-8") as f:
            tokenizer_info = json.load(f)
        vocab_size = tokenizer_info["vocab_size"]

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)

def create_derangement(n):
    """创建一个错位排列，确保没有元素在原位"""
    if n == 1:
        raise ValueError("Cannot create derangement for single element")
    
    import random
    
    # 使用 Sattolo's algorithm - Fisher-Yates shuffle 的变体
    # 保证生成的是错位排列
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        # 选择 [0, i) 范围内的随机索引，不包括 i
        j = random.randint(0, i - 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    return perm

def trim_text(texts: List[str], frac_lengths: torch.Tensor) -> List[str]:
    """
    对文本列表进行裁剪，保持单词/字符边界的完整性
    
    Args:
        texts: 字符串列表
        frac_lengths: 每个文本的裁剪比例 (0.1~0.4)
    
    Returns:
        裁剪后的文本列表
    """
    
    # 使用与 prefix_by_word_boundary_Emilia 相同的分词模式
    word_chars_unicode = (
        "a-zA-Z"         # 基本拉丁字母
        "\u00C0-\u02FF\u1E00-\u1EFF"  # 拉丁字母扩展
        "\u0370-\u03FF\u1F00-\u1FFF"  # 希腊字母
        "\u0400-\u04FF"  # 西里尔字母 
        "\u0590-\u05FF"  # 希伯来语
        "\u0600-\u06FF"  # 阿拉伯语
    )
    other_symbols = "\u2211\u221A\u221E\u2260\u2264\u2192\u20B1\u20B9\u20BD\u2103"
    
    tokenization_pattern = re.compile(
        rf"[{word_chars_unicode}]+'[{word_chars_unicode}]+"  # 带撇号的词
        r"|\d+\.\d+"            # 小数
        r"|\d+-\d+"             # 数字范围
        r"|[a-zA-Z]+-\d+"       # 字母-数字
        r"|\d+-[a-zA-Z]+"       # 数字-字母
        rf"|[{word_chars_unicode}0-9\$€£¢¥%{other_symbols}]+"  # 扩展字符
        r"|[\u3100-\u9fff]"     # 中文字符
        r"|[\u0E00-\u0E7F]"     # 泰文
        r"|[\u3040-\u309F\u30A0-\u30FF]"  # 日语
        r"|[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]"  # 韩文
        r"|\s+"                 # 空格
        r"|\S"                  # 其他单个字符
    )
    
    # 计数单元模式
    countable_pattern = re.compile(
        rf'^[{word_chars_unicode}]+(?:\'[{word_chars_unicode}]+)?$'
        r'|^[\u3100-\u9fff]$'   # 单个中文
        r'|^[\u0E00-\u0E7F]$'   # 单个泰文
        r'|^[\u3040-\u309F\u30A0-\u30FF]$'   # 日语
        r'|^[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]$'   # 韩文
    )
    
    processed_results = []
    
    for i, text in enumerate(texts):
        text = text.strip()
        if not text:
            processed_results.append("")
            continue
        
        # 获取保留比例
        keep_ratio = frac_lengths[i].item()  # 转换为保留比例
        
        # 分词
        tokens = tokenization_pattern.findall(text)
        
        # 计算可计数单元的总数
        countable_units = []
        countable_indices = []
        for idx, token in enumerate(tokens):
            if countable_pattern.fullmatch(token):
                countable_units.append(token)
                countable_indices.append(idx)
        
        if not countable_units:
            # 没有可计数单元，按字符裁剪
            keep_len = max(1, int(len(text) * keep_ratio))
            processed_results.append(text[:keep_len])
            continue
        
        # 计算要保留的可计数单元数
        keep_count = max(5, int(len(countable_units) * keep_ratio))
        
        # 找到最后一个要保留的可计数单元的索引
        if keep_count >= len(countable_units):
            # 保留所有内容
            processed_results.append(text)
            continue
        
        # 找到要保留的最后一个可计数单元在原始tokens中的位置
        last_keep_idx = countable_indices[keep_count - 1]
        
        # 从该位置之后找到下一个非空白字符的位置
        end_idx = last_keep_idx + 1
        while end_idx < len(tokens):
            token = tokens[end_idx]
            # 如果是空白或标点，继续包含
            if not countable_pattern.fullmatch(token):
                # 如果是标点符号，包含它
                if token.strip() and not token.isspace():
                    end_idx += 1
                    break
                end_idx += 1
            else:
                # 遇到下一个可计数单元，停止
                break
        
        # 构建裁剪后的文本
        trimmed_tokens = tokens[:end_idx]
        trimmed_text = "".join(trimmed_tokens).rstrip()
        
        processed_results.append(trimmed_text)
    
    return processed_results