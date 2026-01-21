import dspy

# message history demo

class QA(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()

def simple_history():
    print("-------------------simple history demo")
    dspy.settings.configure(lm=dspy.LM(
        model="ollama_chat/llama3.2:3b",
        model_type="chat",
        base_url="http://localhost:11111",
    ))

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

def few_shot_history():
    print("-------------------few shot history demo")
    dspy.configure(lm=dspy.LM(
        model="openai/llama3.2:3b",
        model_type="chat",
        api_key="ollama",
        base_url="http://localhost:11111/v1",
    ))
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

if __name__ == "__main__":
    simple_history()
    few_shot_history()