import dspy

# 支持多跳搜索、工具调用、react
# 示例中使用了教师模型，用大模型教小模型
# teacher: llama3.1:8b
# small model: llama3.2:3b
# pip install -U bm25s PyStemmer "jax[cpu]"
# 使用BM25S检索算法库：pip install -U bm25s PyStemmer "jax[cpu]"
# 数据：
# 1. 使用2017年wikipedia数据集，包含5,000,000 wiki页面
# 2. 动态从huggingface加载数据

'''
检索算法：
BM25：传统的经典文本检索算法，基于词频和文档频率
BM25S：Sparse BM25，稀疏版BM25，专为高维稀疏向量设计
主要思想：将 BM25 的评分机制转换为稀疏向量表示，使其能与现代稀疏检索技术结合
'''

'''
根据一个声明搜索出wiki中相关内容及标题
1.根据一个声明，生成查询语句
2.根据查询语句在wiki中搜索相关内容作为搜索结果
3.拆分搜索结果拆分中内容、标题，记录有效的内容和标题
4.根据多跳次数设定，循环1、2、3步骤
5.输出所有的内容及标题
'''

# student
lm = dspy.LM('<your_provider>/Llama-3.1-8B-Instruct', max_tokens=3000)
# teacher
gpt4o = dspy.LM('openai/gpt-4o', max_tokens=3000)

dspy.configure(lm=lm)

from dspy.utils import download

if not os.path.exists("wiki.abstracts.2017.jsonl"):
    download("https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz")
    # 解压文件
    # tar -xzvf wiki.abstracts.2017.tar.gz
    import tarfile
    with tarfile.open("wiki.abstracts.2017.tar.gz", "r:gz") as tar:
        tar.extractall()

import orjson
corpus = []

with open("wiki.abstracts.2017.jsonl") as f:
    for line in f:
        line = orjson.loads(line)
        corpus.append(f"{line['title']} | {' '.join(line['text'])}")

len(corpus)

# 检索算法初始化数据索引
import bm25s
import Stemmer

stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25(k1=0.9, b=0.4)
retriever.index(corpus_tokens)

# 从huggingface动态加载数据
import random
from dspy.datasets import DataLoader

kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
hover = DataLoader().from_huggingface(dataset_name="vincentkoc/hover-parquet", split="train", trust_remote_code=True, **kwargs)

hpqa_ids = set()
hover = [
    dspy.Example(claim=x.claim, titles=list(set([y["key"] for y in x.supporting_facts]))).with_inputs("claim")
    for x in hover
    if x["num_hops"] == 3 and x["hpqa_id"] not in hpqa_ids and not hpqa_ids.add(x["hpqa_id"])
]

random.Random(0).shuffle(hover)
trainset, devset, testset = hover[:200], hover[200:500], hover[650:]

example = trainset[0]

print("Claim:", example.claim)
print("Pages that must be retrieved:", example.titles)

def search(query: str, k: int) -> list[str]:
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
    results, scores = retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return run

class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> dspy.Prediction:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))

def top5_recall(example, pred, trace=None):
    gold_titles = example.titles
    recall = sum(x in pred.titles[:5] for x in gold_titles) / len(gold_titles)

    # If we're "bootstrapping" for optimization, return True if and only if the recall is perfect.
    if trace is not None:
        return recall >= 1.0
    
    # If we're just doing inference, just measure the recall.
    return recall
print("-----构建评估器和优化器-----")
evaluate = dspy.Evaluate(devset=devset, metric=top5_recall, num_threads=16, display_progress=True, display_table=5)

models = dict(prompt_model=gpt4o, teacher_settings=dict(lm=gpt4o))
tp = dspy.MIPROv2(metric=top5_recall, auto="medium", num_threads=16, **models)

print("-----评估未优化模型-----")
evaluate(Hop())

print("-----执行模型优化-----")
kwargs = dict(minibatch_size=40, minibatch_full_eval_steps=4)
optimized = tp.compile(Hop(), trainset=trainset, max_bootstrapped_demos=4, max_labeled_demos=4, **kwargs)

print("-----评估已优化模型-----")
evaluate(optimized)

optimized(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles
dspy.inspect_history(n=2)

print("-----加载已优化模型-----")
optimized.save("optimized_hop.json")

loaded_program = Hop()
loaded_program.load("optimized_hop.json")

loaded_program(claim="The author of the 1960s unproduced script written for The Beatles, Up Against It, and Bernard-Marie Koltès are both playwrights.").titles

