import requests
import json
import os

class DatasetLoader:
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def load_dataset(self, url, force_refresh=False):
        """智能加载数据集，支持缓存和重试"""
        
        # 生成缓存文件名
        cache_file = os.path.join(self.cache_dir, "facility_support_dataset.json")
        
        # 如果有缓存且不强制刷新，使用缓存
        if os.path.exists(cache_file) and not force_refresh:
            print("使用缓存文件...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"缓存文件损坏: {e}")
        
        # 尝试不同的方法
        methods = [
            self._method_direct_download,
            self._method_mirror_cdn,
            self._method_multipart_download
        ]
        
        last_error = None
        for method in methods:
            try:
                print(f"尝试方法: {method.__name__}")
                data = method(url)
                
                # 保存到缓存
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"数据已缓存到: {cache_file}")
                
                return data
            except Exception as e:
                print(f"方法失败: {e}")
                last_error = e
                continue
        
        # 如果所有方法都失败，尝试使用缓存（即使过期）
        if os.path.exists(cache_file):
            print("所有方法失败，使用过期缓存...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        raise Exception(f"所有加载方法都失败: {last_error}")
    
    def _method_direct_download(self, url):
        """直接下载方法"""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def _method_mirror_cdn(self, url):
        """使用CDN镜像"""
        # 尝试 jsDelivr CDN
        if "raw.githubusercontent.com" in url:
            # 转换格式: https://raw.githubusercontent.com/user/repo/branch/path
            # 到: https://cdn.jsdelivr.net/gh/user/repo@branch/path
            parts = url.split("/")
            user = parts[3]
            repo = parts[4]
            branch = parts[5]
            path = "/".join(parts[6:])
            
            cdn_url = f"https://cdn.jsdelivr.net/gh/{user}/{repo}@{branch}/{path}"
            print(f"尝试CDN链接: {cdn_url}")
            
            response = requests.get(cdn_url, timeout=10)
            response.raise_for_status()
            return response.json()
        
        raise Exception("无法转换为CDN链接")
    
    def _method_multipart_download(self, url):
        """分块下载（针对大文件）"""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        content = b""
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content += chunk
        
        return json.loads(content.decode('utf-8'))

# 使用
loader = DatasetLoader()
url = "https://raw.githubusercontent.com/meta-llama/llama-prompt-ops/refs/heads/main/use-cases/facility-support-analyzer/dataset.json"

try:
    dataset = loader.load_dataset(url)
    print(f"✅ 数据集加载成功")
    print(f"数据类型: {type(dataset)}")
    if isinstance(dataset, list):
        print(f"数据条数: {len(dataset)}")
        if len(dataset) > 0:
            print(f"第一条数据: {dataset[0]}")
    elif isinstance(dataset, dict):
        print(f"字典键: {list(dataset.keys())}")
except Exception as e:
    print(f"❌ 加载失败: {e}")