import dspy

dspy.configure(lm=dspy.LM(
        model="ollama_chat/llama3.2:3b",
        base_url=f"http://192.168.3.17:11111",
        temperature=0.7))

print("-----------简单问答")
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")
print(response.response)
dspy.inspect_history(n=1)

print("-----------思维链问答")
cot = dspy.ChainOfThought('question -> response')
response = cot(question="should curly braces appear on their own line?")
print(response.response)
dspy.inspect_history(n=1)

print("-----------思维链问答1")
cot = dspy.ChainOfThought('question -> response')
response = cot(question="what are high memory and low memory on linux?")
print(response.response)
dspy.inspect_history(n=1)