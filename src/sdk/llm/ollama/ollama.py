import dspy
import requests
import json
import ollama
# from ollama import Client

# IP_ADDR = "localhost"
IP_ADDR = "192.168.3.17"

# 创建Ollama模型适配器
class OllamaLM(dspy.BaseLM):
    def __init__(self, model="llama3", base_url=f"http://{IP_ADDR}:11111", max_tokens=500, temperature=0.7, **kwargs):
        self.ollama_client = ollama.Client(host=base_url)

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        # 注意：必须指定 model_type="chat" 才能触发正确的适配器行为
        super().__init__(model=model, model_type="chat")  # 关键！设为 chat 模型

    def configure(self, **kwargs):
        if "model_type" in kwargs:
            self.model_type = kwargs["model_type"]

    def basic_request(self, messages, **kwargs):
        print("-------------------basic_request")
        kwargs = {**self.kwargs, **kwargs}
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

    def chat(self, messages, **kwargs):
        print("-------------------chat")
        kwargs = {**self.kwargs, **kwargs}
        # 将 messages 转换为 Ollama 格式        
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        # 调用 Ollama API
        response = self.ollama_client.chat(
            model=self.model,
            messages=ollama_messages,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        )
        print("-------------------chat response")
        print(response)
        print("-------------------")

        # 提取助手回复
        message = response["message"]
        # 检查message类型：如果是Message对象，使用属性访问；如果是字典，使用键访问
        if hasattr(message, "role"):
            # 单个Message对象
            if message.role == "assistant":
                return message.content
        elif isinstance(message, dict):
            # 单个字典
            if message["role"] == "assistant":
                return message["content"]
        elif isinstance(message, list):
            # 字典列表
            assistant_messages = [
                msg for msg in message if msg.get("role") == "assistant"
            ]
            return assistant_messages[0]["content"] if assistant_messages else ""
        return ""

    def __call__(self, messages, **kwargs):
        # DSPy 会传入 messages（list of dict），不是 prompt 字符串
        is_chat = self.model_type == "chat"
        if is_chat:
            response = self.chat(messages, **kwargs)
            # 返回 list of completions（DSPy 要求）
            return [response.strip()]
        else:
            response = self.basic_request(messages, **kwargs)
            # 返回 list of completions（DSPy 要求）
            return [response["response"].strip()]

