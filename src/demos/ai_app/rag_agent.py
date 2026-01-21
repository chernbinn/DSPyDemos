import random
import dspy
from dspy.datasets import DataLoader
from typing import Literal

'''
1.根据一个声明，使用工具在wiki中找到一致或者反驳的页面标题
2.多跳搜索
3.教师-学生模型，教师生成数据后训练学生
'''

# 支持多跳搜索、工具调用、react
# 示例中使用了教师模型，用大模型教小模型
# teacher: llama3.1:8b
# small model: llama3.2:3b
llama3b = dspy.LM('ollama_chat/llama3.2:3b', temperature=0.7)
gpt4o = dspy.LM('ollama_chat/llama3.1:8b', temperature=0.7)

dspy.configure(lm=llama3b)

kwargs = dict[str, tuple[Literal['claim'], Literal['supporting_facts'], Literal['hpqa_id'], Literal['num_hops']] | tuple[Literal['claim']]](fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="vincentkoc/hover-parquet", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:100], hover[100:200], hover[650:]

example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

DOCS = {}

# 工具
def search(query: str, k: int) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=k)
    results = [x['text'] for x in results]

    for result in results:
        title, text = result.split(" | ", 1)
        DOCS[title] = text

    return results

def search_wikipedia(query: str) -> list[str]:
    """Returns top-5 results and then the titles of the top-5 to top-30 results."""

    topK = search(query, 30)
    titles, topK = [f"`{x.split(' | ')[0]}`" for x in topK[5:30]], topK[:5]
    return topK + [f"Other retrieved pages have titles: {', '.join(titles)}."]

def lookup_wikipedia(title: str) -> str:
    """Returns the text of the Wikipedia page, if it exists."""

    if title in DOCS:
        return DOCS[title]

    results = [x for x in search(title, 10) if x.startswith(title + " | ")]
    if not results:
        return f"No Wikipedia page found for title: {title}"
    return results[0]
# tools end

instructions = "Find all Wikipedia titles relevant to verifying (or refuting) the claim."
signature = dspy.Signature("claim -> titles: list[str]", instructions)
react = dspy.ReAct(signature, tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
# 使用一次模型调用
print("-----调用模型-----")
react(claim="David Gregory was born in 1625.").titles[:3]

# 评估指标：top-5 recall
def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0
    
    # If we're just doing inference, just measure the recall.
    return recall
# 构建评估器
evaluate = dspy.Evaluate(devset=devset, metric=top5_recall, num_threads=16, display_progress=True, display_table=5)
# 根据评估稳定性的需求，优化模型调用结果
def safe_react(claim: str):
    try:
        return react(claim=claim)
    except Exception as e:
        return dspy.Prediction(titles=[])
# 评估模型
print("-----评估未优化模型-----")
evaluate(safe_react)

# ----------- 优化模型及评估优化后模型 -----------
kwargs = dict(teacher_settings=dict(lm=gpt4o), prompt_model=gpt4o, max_errors=999)
# 初始化优化指标函数
tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **kwargs)

print("-----执行模型优化-----")
# 构建优化器
optimized_react = tp.compile(react, trainset=trainset, max_bootstrapped_demos=3, max_labeled_demos=0)

print("-----评估优化后的模型-----")
# 评估优化后的模型
evaluate(optimized_react)
print("-----调用优化后的模型-----")
# 调用优化后的模型
optimized_react(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
dspy.inspect_history(n=2)

# 保存优化后的模型数据
optimized_react.save("optimized_react.json")

# ----------- 加载优化后的模型数据并调用 -----------
print("Loaded optimized_react.json")
# 创建模型调用实例
loaded_react = dspy.ReAct("claim -> titles: list[str]", tools=[search_wikipedia, lookup_wikipedia], max_iters=20)
# 加载优化后的模型数据
loaded_react.load("optimized_react.json")

print("-----调用加载优化数据后的模型-----")
# 调用加载优化数据后的模型
loaded_react(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles