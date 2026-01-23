import requests

# 尝试不同的镜像源
urls = [
    # 原始链接
    "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json",
    # 使用 jsDelivr CDN（如果仓库是公开的）
    "https://cdn.jsdelivr.net/gh/meta-llama/llama-prompt-ops@main/use-cases/facility-support-analyzer/dataset.json",
    # 使用 GitHub Pages（如果有）
    # "https://meta-llama.github.io/llama-prompt-ops/use-cases/facility-support-analyzer/dataset.json",
]

def try_multiple_urls(urls):
    """尝试多个URL直到成功"""
    for url in urls:
        try:
            print(f"尝试: {url}")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"成功从 {url} 加载")
                return response.json()
        except Exception as e:
            print(f"{url} 失败: {e}")
            continue
    raise Exception("所有URL尝试都失败")

# 使用
try:
    dataset = try_multiple_urls(urls)
except Exception as e:
    print(f"所有方法都失败: {e}")