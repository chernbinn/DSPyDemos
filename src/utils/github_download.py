import requests
import json
import time

def load_dataset_with_retry(url, max_retries=3):
    """带重试机制的数据集加载"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, 
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            print(f"尝试 {attempt+1}/{max_retries} 失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception(f"无法加载数据集，已重试{max_retries}次")

# 使用
url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"
try:
    dataset = load_dataset_with_retry(url)
    print(f"数据集加载成功，大小: {len(dataset)}")
except Exception as e:
    print(f"加载失败: {e}")