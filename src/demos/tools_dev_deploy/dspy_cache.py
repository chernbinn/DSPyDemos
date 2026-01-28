
'''
在这个教程中，我们将探讨DSPy的缓存机制的设计，并演示如何有效地使用和定制它。

DSPy的缓存系统分为三个独立层次：

内存缓存：使用cachetools.LRUCache实现，该层为频繁使用的数据提供快速访问。
磁盘缓存：利用diskcache.FanoutCache，该层为缓存项提供持久化存储。
提示缓存（服务器端缓存）：由LLM服务提供商（例如OpenAI、Anthropic）管理。
虽然DSPy不直接控制服务器端提示缓存，但它允许用户根据具体需求启用、禁用和自定义内存缓存和磁盘缓存。

默认情况下，在DSPy中，内存和磁盘缓存都会自动启用。无需执行特定操作即可开始使用缓存。当发生缓存命中时，
您将观察到模块调用执行时间的显著减少。此外，如果启用了使用跟踪，则缓存的调用的使用指标将为None。
'''
import dspy
import os
import time
import dspy
from typing import Dict, Any, Optional
import orjson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "ollama"
lm=dspy.LM(
    model="openai/llama3.2:3b",
    base_url="http://localhost:11111/v1"
)
dspy.configure(lm=lm, track_usage=True)

def dspy_cache():
    predict = dspy.Predict("question->answer")

    start = time.time()
    result1 = predict(question="Who is the GOAT of basketball?")
    print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

    start = time.time()
    result2 = predict(question="Who is the GOAT of basketball?")
    print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")

'''
使用提供方提示缓存¶ 除了DSPy的内置缓存机制外，您还可以利用Anthropic和OpenAI等LLM提供方提供的提供方提示缓存。
当与dspy.ReAct()等发送重复相似提示的模块一起使用时，此功能特别有用，因为它通过在提供方的服务器上缓存提示前
缀来减少延迟和成本。

通过给dspy.LM()传递cache_control_injection_points参数，启动提供方提示缓存。比如，Anthropic和OpenAI
端会支持该功能。详细情况，参考LiteLLM prompt caching documentation
（https://docs.litellm.ai/docs/tutorials/prompt_caching#configuration），也即支持该功能需要是基于
litellm的lm提供方。
特别是以下情况比较有用：
1.使用dspy.ReAct()模块，因为它们会发送重复的提示。
2.处理保持不变的长系统提示。
3.相似的上下文中处理多个请求。
'''
def provider_cache():
    lm = dspy.LM(
        model="openai/llama3.2:3b",
        base_url="http://localhost:11111/v1",
        cache_control_injection_points=[
            {
                "location": "message",
                "role": "system",
            }
        ],
    )
    dspy.configure(lm=lm)

    # Use with any DSPy module
    predict = dspy.Predict("question->answer")
    result = predict(question="What is the capital of France?")
    print(result)
    print(result.get_lm_usage())

'''
在某些情况下，需要关闭全部或者部分缓存能力，缓存能力有内存缓存、磁盘缓存。
比如：
1.相同的请求需要有不同的回复。
2.没有磁盘写权限，需要关闭磁盘缓存。
3.有限的内存控件，需要关闭内存缓存。
'''
def disable_cache():
    print('''
# 关闭全部缓存能力
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# 管理缓存大小
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
    ''')

'''
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {write your own constructor}

    def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {write your logic of computing cache key}

    def get(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {write your cache read logic}

    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {write your cache write logic}
'''

def call_lm_with_cache():
    dspy.configure(lm=lm)

    predict = dspy.Predict("question->answer")

    start = time.time()
    result1 = predict(question="Who is the GOAT of soccer?")
    print(f"Time elapse: {time.time() - start: 2f}")

    start = time.time()
    with dspy.context(lm=lm):
        result2 = predict(question="Who is the GOAT of soccer?")
    print(f"Time elapse: {time.time() - start: 2f}")

def call_lm_without_cache():
    dspy.configure(lm=lm)   

    class CustomCache(dspy.clients.Cache):

        def cache_key(self, request: dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
            messages = request.get("messages", [])
            return sha256(orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)).hexdigest()

    dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

    predict = dspy.Predict("question->answer")

    start = time.time()
    result1 = predict(question="Who is the GOAT of volleyball?")
    print(f"Time elapse: {time.time() - start: 2f}")

    start = time.time()
    with dspy.context(lm=lm):
        result2 = predict(question="Who is the GOAT of volleyball?")
    print(f"Time elapse: {time.time() - start: 2f}")

if __name__ == "__main__":
    dspy_cache()
    provider_cache()
    disable_cache()
    call_lm_with_cache()
    call_lm_without_cache()