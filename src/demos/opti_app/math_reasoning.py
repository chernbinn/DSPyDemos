import dspy
from dspy.datasets import MATH

'''
本demo演示了优化器如何通过寻找更好的提示策略和少量示例来显著提高复杂数学问题的性能。
'''

# pip install git+https://github.com/hendrycks/math.git
# 安装 Hendrycks 的 MATH 数据集和相关工具，主要用于数学问题求解和评估

# 学生模型
gpt4o_mini = dspy.LM('ollama_chat/llama3.2:3b', api_key='sk-', max_tokens=2000, base_url='http://localhost:11111')
# 老师模型
gpt4o = dspy.LM('ollama_chat/ministral-3:3b', api_key='sk-', max_tokens=2000, base_url='http://localhost:11111')
dspy.configure(lm=gpt4o_mini)  # we'll use gpt-4o-mini as the default LM, unless otherwise specified

dataset = MATH(subset='algebra')
print("train set size:", len(dataset.train), "test set size:", len(dataset.dev))
print("-----数据示例：")
example = dataset.train[0]
print("Question:", example.question)
print("Answer:", example.answer)

module = dspy.ChainOfThought("question -> answer")
print("\n-----调用未优化模型-----")
print(module(question=example.question))
print("-----查看提示-----")
dspy.inspect_history(n=1)

THREADS = 10
kwargs = dict(num_threads=THREADS, display_progress=True, display_table=5)
evaluate = dspy.Evaluate(devset=dataset.dev, metric=dataset.metric, **kwargs)
print("-----评估未优化模型-----")
#print(evaluate(dspy.ChainOfThought("question -> answer")))
print(evaluate(module))

print("-----构建教师优化器")
kwargs = dict(num_threads=THREADS, teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o_mini)
optimizer = dspy.MIPROv2(metric=dataset.metric, auto="medium", **kwargs)

kwargs = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
print("-----执行教师优化-----")
optimized_module = optimizer.compile(module, trainset=dataset.train, **kwargs)

print("-----评估优化后的模型-----")
print(evaluate(optimized_module))

print("-----查看优化后的提示-----")
dspy.inspect_history()
