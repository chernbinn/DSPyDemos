'''
本教程演示了如何使用CoNLL-2003数据集和DSPy进行实体提取。重点是提取指代人的实体。我们将：
从CoNLL-2003数据集中提取和标记指代人的实体
定义一个用于提取指代人实体的DSPy程序
在CoNLL-2003数据集的子集上优化和评估该程序
'''
import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy

# 加载和准备数据
# 1. 从huggingface加载数据
# 2. 定义一个函数来提取指代人物的标记
# 3. 将数据集切片以创建用于训练和测试的小型子集。
def load_conll_dataset() -> dict:
    """
    Loads the CoNLL-2003 dataset into train, validation, and test splits.
    
    Returns:
        dict: Dataset splits with keys 'train', 'validation', and 'test'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a temporary Hugging Face cache directory for compatibility with certain hosted notebook
        # environments that don't support the default Hugging Face cache directory
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("conll2003", trust_remote_code=True)

def extract_people_entities(data_row: dict[str, Any]) -> list[str]:
    """
    Extracts entities referring to people from a row of the CoNLL-2003 dataset.
    
    Args:
        data_row (dict[str, Any]): A row from the dataset containing tokens and NER tags.
    
    Returns:
        list[str]: List of tokens tagged as people.
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in (1, 2)  # CoNLL entity codes 1 and 2 refer to people
    ]

def prepare_dataset(data_split, start: int, end: int) -> list[dspy.Example]:
    """
    Prepares a sliced dataset split for use with DSPy.
    
    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.
    
    Returns:
        list[dspy.Example]: List of DSPy Examples with tokens and expected labels.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_people=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

# Load the dataset
dataset = load_conll_dataset()

# Prepare the training and test sets
train_set = prepare_dataset(dataset["train"], 0, 50)
test_set = prepare_dataset(dataset["test"], 0, 200)

# 创建一个提取器程序
from typing import List

class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")

people_extractor = dspy.ChainOfThought(PeopleExtraction)
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 定义评估指标和初始化评估函数
def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    Computes correctness of entity extraction predictions.
    
    Args:
        example (dspy.Example): The dataset example containing expected people entities.
        prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
        trace: Optional trace object for debugging.
    
    Returns:
        bool: True if predictions match expectations, False otherwise.
    """
    return prediction.extracted_people == example.expected_extracted_people

evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=extraction_correctness_metric,
    num_threads=24,
    display_progress=True,
    display_table=True
)
# 在测试数据上进行提取器性能评估
evaluation_result = evaluate_correctness(people_extractor, devset=test_set)
print("Initial Extractor Performance:", evaluation_result)

# 使用MIPROv2算法优化提取器在数据上的表现
mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    auto="medium",
)
optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    minibatch=False
)
# 在测试数据上评估优化后的提取器性能
evaluation_result = evaluate_correctness(optimized_people_extractor, devset=test_set)
print("Optimized Extractor Performance:", evaluation_result)
dspy.inspect_history(n=1)
optimized_people_extractor.save("optimized_extractor.json")

# 加载优化后的提取器
loaded_people_extractor = dspy.ChainOfThought(PeopleExtraction)
loaded_people_extractor.load("optimized_extractor.json")

loaded_people_extractor(tokens=["Italy", "recalled", "Marcello", "Cuttitta"]).extracted_people