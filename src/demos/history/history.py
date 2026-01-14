import dspy
from sdk.llm.ollama.ollama import OllamaLM

def ollama():
    # 创建并配置Ollama模型实例
    ollama_lm = OllamaLM(model="llama3.2:3b")
    # print(ollama_lm("你好"))
    ollama_lm.configure(model_type="chat")
    return ollama_lm

dspy.settings.configure(lm=ollama())

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

predict = dspy.Predict(QA)
history = dspy.History(messages=[])

while True:
    question = input("Type your question, end conversation by typing 'finish': ")
    if question == "finish":
        break
    outputs = predict(question=question, history=history)
    print(f"\n{outputs.answer}\n")
    history.messages.append({"question": question, **outputs})

print("-------------------history")
dspy.inspect_history()