

import dspy
from dspy import GEPA
from datasets import load_dataset

'''
基于反思算法优化数学问题求解模型
'''
# api_key = input("Enter your OpenAI API key: ")

lm = dspy.LM("openai/qwen3-vl:4b", temperature=1, api_key="sk-", max_tokens=32000, base_url="https://localhost:11111/v1")
reflection_lm = dspy.LM(model="openai/qwen3-vl:8b", temperature=1.0, max_tokens=32000, api_key="sk-", base_url="https://localhost:11111/v1")
dspy.configure(lm=lm)

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")['train']
    print("----- 成功加载AIME 2025测试集 -----")
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]

    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    # 由于是小数量测试集，这里简单重复5次
    test_set = test_split * 5

    return train_set, val_set, test_set

print("----- 初始化数据集 -----")
train_set, val_set, test_set = init_dataset()

print(f"train_set: {len(train_set)}, val_set: {len(val_set)}, test_set: {len(test_set)}")

print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)

def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

print("----- 评估模型 -----")
print(evaluate(program))

# EvaluationResult(score=46.67, results=<list of 150 results>)

def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."
    
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=reflection_lm, # 反思模型的机制需要深入代码理解。反思模型的机制是基于语言模型的文本反馈，而不是标量指标。该模型一般需要更为智能。
)

print("----- 使用反思算法优化模型 -----")
optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

print(optimized_program.predict.signature.instructions)

print("----- 评估优化后的模型 -----")
evaluate(optimized_program)
# EvaluationResult(score=56.67, results=<list of 150 results>)