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

predict.demos.append(
    dspy.Example(
        question="What is the capital of France?",
        history=dspy.History(
            messages=[{"question": "What is the capital of Germany?", "answer": "The capital of Germany is Berlin."}]
        ),
        answer="The capital of France is Paris.",
    )
)

# predict(question="What is the capital of America?", history=dspy.History(messages=[]))
predict(question="中国的首都在哪里？", history=dspy.History(messages=[]))
dspy.inspect_history()