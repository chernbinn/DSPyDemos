import dspy
import requests
import json

# 创建Ollama模型适配器
class OllamaLM(dspy.LM):
    def __init__(self, model="llama3", base_url="http://localhost:11111", max_tokens=500, temperature=0.7, **kwargs):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        # 注意：必须指定 model_type="chat" 才能触发正确的适配器行为
        super().__init__(model=model, model_type="chat")  # 关键！设为 chat 模型

    def basic_request(self, messages, **kwargs):
        # 将 messages 转换为纯文本 prompt（简单拼接）
        # 更高级的做法可用 tokenizer 或模板，这里简化处理
        prompt = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in messages
        ) + "\nAssistant:"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

        result = response.json()
        # 保存历史（用于 inspect_history）
        self.history.append({
            "messages": messages,
            "prompt": prompt,
            "response": result,
            "kwargs": kwargs,
        })
        return result

    def __call__(self, messages, **kwargs):
        # DSPy 会传入 messages（list of dict），不是 prompt 字符串
        response = self.basic_request(messages, **kwargs)
        # 返回 list of completions（DSPy 要求）
        return [response["response"].strip()]

# 简单测试
class BasicQA(dspy.Signature):
    """回答以下问题"""
    question = dspy.InputField()
    answer = dspy.OutputField()


def ollama1():
    # 创建并配置Ollama模型实例
    ollama_lm = OllamaLM(model="llama3.2:3b")
    # print(ollama_lm("你好"))

    dspy.settings.configure(lm=ollama_lm)

def cli():    
    ollama1()
    qa = dspy.Predict(BasicQA)
    response = qa(question="巴黎是哪个国家的首都？")
    print(response.answer)

if __name__ == "__main__":
    cli()