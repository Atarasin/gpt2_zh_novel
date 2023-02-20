from pynvml import *
from typing import Union
from pathlib import Path
from math import floor


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def data_split(data_dir: Union[str, Path], output_dir: Union[str, Path], train_size: float = 0.85):
    """
    读取指定目录的所有txt文件, 按照一定比例划分为train, test
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise Exception('data_dir is not a dictory')

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise Exception('output_dir is not a dictory')

    lines = []

    print("------------------开始读取txt文件--------------------")

    for file in data_dir.glob('*.txt'):
        with file.open('r', encoding='utf-8') as f:
            lines.extend(f.readlines())

    print("------------------txt文件读取完成--------------------")

    total_lines_num = len(lines)
    total_train_lines_num = floor(total_lines_num * train_size)

    # create 'train.txt' and 'test.txt'
    train_file = Path(output_dir, 'train.txt')
    test_file = Path(output_dir, 'test.txt')

    print("------------------开始写入train.txt文件--------------------")

    with train_file.open('w', encoding='utf-8') as f:
        f.writelines(lines[:total_train_lines_num])
    
    print("------------------写入train.txt文件完成--------------------")

    print("------------------开始写入test.txt文件--------------------")
    
    with test_file.open('w', encoding='utf-8') as f:
        f.writelines(lines[total_train_lines_num:])

    print("------------------写入test.txt文件完成--------------------")


if __name__ == "__main__":
    data_split('raw', 'data', train_size=1.0)