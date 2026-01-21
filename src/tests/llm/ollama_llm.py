import dspy
from sdk.llm.ollama.ollama import OllamaLM

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
    return ollama_lm

def demo1():
    ollama_lm = ollama1()
    ollama_lm.configure(model_type="generate")
    qa = dspy.Predict(BasicQA)
    response = qa(question="巴黎是哪个国家的首都？")
    print(response.answer)

def demo2():
    ollama_lm = ollama1()
    ollama_lm.configure(model_type="chat")
    qa = dspy.ChainOfThought(BasicQA)
    response = qa(question="巴黎是哪个国家的首都？")
    print(response.answer)

def cli():
    # demo1()
    demo2()

if __name__ == "__main__":
    cli()