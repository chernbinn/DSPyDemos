from huggingface_hub import hf_hub_download
import os
import requests
import time

def download_by_proxy():
    # 设置代理（如果需要）
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

    try:
        file_path = hf_hub_download(
            repo_id="dspy/cache",
            filename="ragqa_arena_tech_examples.jsonl",
            repo_type="dataset"
        )
        print(f"下载成功: {file_path}")
    except Exception as e:
        print(f"下载失败: {e}")

def download_by_mirror():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    file_path = hf_hub_download(
        repo_id='dspy/cache',
        filename='ragqa_arena_tech_examples.jsonl',
        repo_type='dataset'
    )
    print(f'下载成功: {file_path}')

def download_huggingface_file(repo_id, filename):
    # filename = "ragqa_arena_tech_examples.jsonl"
    # repo_id = "dspy/cache"
    # filename = "ragqa_arena_tech_corpus.jsonl"
    # repo_id = "dspy/cache"
    # repo_id="vincentkoc/hover-parquet",
    # filename="train-00000-of-00001.parquet"
    
    # 尝试的镜像列表（增加更多镜像源）
    mirrors = [
        "https://hf-mirror.com",
        "https://huggingface.co",
        "https://hf.co"
    ]
   
    # 方案：直接下载
    print("\n尝试方案: 直接下载...")
    for mirror in mirrors:
        try:
            url = f"{mirror}/{repo_id}/resolve/main/{filename}"
            print(f"   尝试URL: {url}")
            response = requests.get(url, timeout=60, stream=True)  # 增加超时时间并使用流式下载
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ 成功下载: {filename}")
                return filename
            else:
                print(f"   ❌ 状态码错误: {response.status_code}")
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:100]}...")
            time.sleep(2)  # 每次失败后暂停2秒

    # 方案1：使用 huggingface_hub + 镜像
    print("尝试方案: huggingface_hub + 镜像...")
    for mirror in mirrors:
        try:
            os.environ['HF_ENDPOINT'] = mirror
            print(f"   尝试镜像: {mirror}")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset"
            )
            print(f"✅ 成功通过镜像 {mirror} 下载")
            return file_path
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:100]}...")
            time.sleep(2)  # 每次失败后暂停2秒
    
    print("\n❌ 所有方法都失败了")
    print("建议检查网络连接或尝试使用代理")
    return None

class HGFile:
    def __init__(self, repo_id, filename):
        self.repo_id = repo_id
        self.filename = filename

def download_huggingface_files():
    files = [
        # HGFile("vincentkoc/hover-parquet", "train-00000-of-00001.parquet"),
        HGFile("dspy/cache", "ragqa_arena_tech_examples.jsonl"),
        HGFile("dspy/cache", "ragqa_arena_tech_corpus.jsonl"),        
    ]
    for file in files:
        print(f"下载文件: 从 {file.repo_id} 下载 {file.filename} ")
        if os.path.exists(file.filename):
            print(f"文件已存在: {file.filename}")
            continue
        download_huggingface_file(file.repo_id, file.filename)

if __name__ == '__main__':
    # download_by_proxy()
    # download_by_mirror()
    download_huggingface_files()
