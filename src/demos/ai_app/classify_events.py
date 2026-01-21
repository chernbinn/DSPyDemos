import dspy
from typing import Literal
import json
import pandas as pd
import csv

# 对历史事件进行分类
# 分别使用大模型和小模型进行分离及对比
# 大模型：llama3.3
# 小模型：llama3.2:1b

lm32 = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434')
lm33 = dspy.LM('ollama_chat/llama3.3', api_base='http://localhost:11434')
dspy.configure(lm=lm32)

class Categorize(dspy.Signature):
    """Classify historic events."""

    event: str = dspy.InputField()
    category: Literal[
        "Wars and Conflicts",
        "Politics and Governance",
        "Science and Innovation",
        "Cultural and Artistic Movements",
        "Exploration and Discovery",
        "Economic Events",
        "Social Movements",
        "Man-Made Disasters and Accidents",
        "Natural Disasters and Climate",
        "Sports and Entertainment",
        "Famous Personalities and Achievements"
    ] = dspy.OutputField()
    confidence: float = dspy.OutputField()

# 简单分类器
classify = dspy.Predict(Categorize)

# Define a function to classify the event description
def classify_event(description):
    try:
        prediction = classify(event=description)
        return prediction.category, prediction.confidence
    except Exception as e:
        return 0, 0

# ------------ 评估大模型分类结果 --------------
# 指标函数，用于评估结果
def validate_category(example, prediction, trace=None):
    return prediction.category == example.category

# ------------ 优化分类器 -----------------
dspy.configure(lm=dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434'))

def load_trainset():
    # Load the trainset
    trainset = []
    with open('llama_trainset.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = dspy.Example(event=row['description'], category=row['category']).with_inputs("event")
            trainset.append(example)
    return trainset

def load_testset():
    # Load the testset
    testset = []
    with open('llama_testset.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = dspy.Example(event=row['description'], category=row['category']).with_inputs("event")
            testset.append(example)
    return testset

def non_optimize():
    with open("0101_events.json", 'r') as file:
        data = json.load(file)
        events = pd.DataFrame(data['events'])
        
        print("classifying events using non-optimized lm32:1b...")
        # Using our small model
        with dspy.context(lm=dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434')):
            events['category_32_1b'], events['confidence_32_1b'] = zip(*events['description'].apply(classify_event))

        # Using our large model
        print("classifying events using non-optimized lm33...")
        with dspy.context(lm=dspy.LM('ollama_chat/llama3.3', api_base='http://localhost:11434')):
            events['category_33'], events['confidence_33'] = zip(*events['description'].apply(classify_event))

        events.to_csv("model_compare.csv", index=False)

    # Evaluate our existing function
    print("------- evaluate non-optimized lm32 -------")
    dspy.configure(lm=lm32)
    evaluator = dspy.Evaluate(devset=load_testset(), num_threads=1, display_progress=True, display_table=5)
    print("non-optimized lm32 evaluate result:", evaluator(classify, metric=validate_category))

    print("------- evaluate non-optimized lm33 -------")
    dspy.configure(lm=lm33)
    evaluator = dspy.Evaluate(devset=load_testset(), num_threads=1, display_progress=True, display_table=5)
    print("non-optimized lm33 evaluate result:", evaluator(classify, metric=validate_category))

def general_optimize():
    # Load the trainset
    trainset = []
    with open('llama_trainset.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = dspy.Example(event=row['description'], category=row['category']).with_inputs("event")
            trainset.append(example)

    print("------- general optimize lm32 -------")
    dspy.configure(lm=lm32)
    # 一般优化方式
    tp = dspy.MIPROv2(metric=validate_category, auto="light")
    optimized_classify = tp.compile(classify, trainset=load_trainset(), max_labeled_demos=0, max_bootstrapped_demos=0)

    print("------- evaluate general optimized lm32 -------")
    # Evaluate our existing function
    evaluator = dspy.Evaluate(devset=load_testset(), num_threads=1, display_progress=True, display_table=5)
    print("general optimized lm32 evaluate result:", evaluator(optimized_classify, metric=validate_category))

    optimized_classify.save("general_optimized_classify.json")

def teacher_optimize():
    # 教师优化方式
    # Load our model
    # student model
    lm = lm32
    # teacher model
    prompt_gen_lm = lm33

    dspy.configure(lm=lm)
    # Optimize
    tp = dspy.MIPROv2(metric=validate_category, auto="light", prompt_model=prompt_gen_lm, task_model=lm)
    optimized_classify = tp.compile(classify, trainset=load_testset(), max_labeled_demos=0, max_bootstrapped_demos=0)
    # Evaluate our existing function
    evaluator = dspy.Evaluate(devset=load_testset(), num_threads=1, display_progress=True, display_table=5)
    print("teacher optimized lm32 evaluate result:", evaluator(optimized_classify, metric=validate_category))

    optimized_classify.save("teacher_optimized_classify.json")

def main():
    non_optimize()
    general_optimize()
    teacher_optimize()

if __name__ == "__main__":
    main()

