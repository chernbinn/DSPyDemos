import dspy
import random
import orjson
import os, sys
import argparse
from dspy.utils import download
from dspy.evaluate import SemanticF1

dspy.configure(lm=dspy.LM(
        model="ollama_chat/llama3.2:3b",
        base_url=f"http://192.168.3.17:11111",
        temperature=0.7))

example = None
metric = None
trainset, devset, testset = None, None, None
evaluate_fn = None
def evaluate_init():
    global example, metric, trainset, devset, testset, evaluate_fn
    # Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
    if not os.path.exists("ragqa_arena_tech_examples.jsonl"):
        print("Downloading ragqa_arena_tech_examples.jsonl...")
        download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")

    data = []  
    with open("ragqa_arena_tech_examples.jsonl") as f:
        data = [orjson.loads(line) for line in f]

    # Inspect one datapoint.
    print(data[0])

    data = [dspy.Example(**d).with_inputs('question') for d in data]
    # Let's pick an `example` here from the data.
    example = data[2]
    print(example)

    random.Random(0).shuffle(data)
    trainset, devset, testset = data[:200], data[200:500], data[500:1000]
    print(f"trainset: {len(trainset)}, devset: {len(devset)}, testset: {len(testset)}")

    # Instantiate the metric.
    metric = SemanticF1(decompositional=True)
    # Define an evaluator that we can re-use.
    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24,
                            display_progress=True, display_table=2)
    evaluate_fn = evaluate

evaluate_init()
# ----------------- simple RAG
def simple_rag():
    '''llm直接回复问题，不基于知识内容，回复质量取决于模型能力'''
    print("----- simple RAG -----")
    cot = dspy.ChainOfThought('question -> response')
    # Produce a prediction from our `cot` module, using the `example` above as input.
    pred = cot(**example.inputs())
    # Compute the metric score for the prediction.
    score = metric(example, pred)
    print(f"Question: \t {example.question}\n")
    print(f"Gold Response: \t {example.response}\n")
    print(f"Predicted Response: \t {pred.response}\n")
    print(f"Semantic F1 Score: {score:.2f}")
    dspy.inspect_history(n=1)    

    # Evaluate the Chain-of-Thought program.
    evaluate_fn(cot)

# ----------------- RAG with retrieval
def load_data():
    # Download question--answer pairs from the RAG-QA Arena "Tech" dataset.
    if not os.path.exists("ragqa_arena_tech_corpus.jsonl"):
        print("Downloading ragqa_arena_tech_corpus.jsonl...")
        download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_corpus.jsonl")

    max_characters = 6000  # for truncating >99th percentile of documents
    topk_docs_to_retrieve = 5  # number of documents to retrieve per search query

    with open("ragqa_arena_tech_corpus.jsonl") as f:
        corpus = [orjson.loads(line)['text'][:max_characters] for line in f]
        print(f"Loaded {len(corpus)} documents. Will encode them below.")

    # 在不设置api_base情况下，默认调用open的线上服务，没有购买该服务的情况，无法使用
    # embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
    # 基于本地ollama模型调用兼容openai的embed接口，可以调用成功
    # openai/quentinz/bge-base-zh-v1.5:q8_0，模型上下文太小，运行失败
    # openai/bge-m3:567m：向量化结果存在特殊符号，不能被正确处理
    # openai/embeddinggemma:300m，模型可用
    # 总结：相同的内容，不同的模型向量化结果不同，需要根据模型能力选择合适的模型
    embedder = dspy.Embedder('openai/embeddinggemma:300m', 
                    api_key="sk-ollama",
                    api_base=f"http://192.168.3.17:11111/v1")
    '''
    # 使用ollama的非openai格式调用模型，在litellm中会出现post请求变成get请求，导致调用失败。
    # 逻辑上litellm是正确的，HTTPHandler中post请求会被转换为get请求，但是ollama的embed接口不支持get请求
    embedder = dspy.Embedder('ollama/quentinz/bge-base-zh-v1.5:q8_0', 
                    api_base=f"http://192.168.3.17:11111/")
    '''
    return embedder, corpus, topk_docs_to_retrieve

embedder, corpus, topk_docs_to_retrieve = load_data()
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)

class RAG(dspy.Module):
    '''基于知识内容回复问题，回复质量取决于模型能力和知识内容'''
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)

def optimized_rag():
    print("----- optimized RAG -----")
    load_data()
    rag = RAG()
    rag(question="what are high memory and low memory on linux?")
    dspy.inspect_history()

    print("----- evaluate RAG -----")
    evaluate_fn(RAG())

    tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)  # use fewer threads if your rate limit is small
    print("----- compile RAG -----")
    optimized_rag = tp.compile(RAG(), trainset=trainset,
                            max_bootstrapped_demos=2, max_labeled_demos=2)

    print("----- 一次普通调用 ----")
    baseline = rag(question="cmd+tab does not work on hidden or minimized windows")
    print(baseline.response)

    print("----- 一次优化后的调用 ----")
    pred = optimized_rag(question="cmd+tab does not work on hidden or minimized windows")
    print(pred.response)
    dspy.inspect_history(n=2)
    print("----- evaluate optimized RAG -----")
    evaluate_fn(optimized_rag)

    optimized_rag.save("optimized_rag.json")

def loaded_rag():
    print("----- loaded RAG -----")
    loaded_rag = RAG()
    loaded_rag.load("optimized_rag.json")

    loaded_rag(question="cmd+tab does not work on hidden or minimized windows")

def run_rag(args):
    if args.optimized:
        optimized_rag()
    if args.load:
        loaded_rag()
    else:
        simple_rag()

if __name__ == "__main__":    
    # 解析参数
    parser = argparse.ArgumentParser(description="RAG 程序")
    parser.add_argument("--optimized", action="store_true", help="使用优化后的 RAG")
    parser.add_argument("--load", action="store_true", help="加载已保存的 RAG")

    args = parser.parse_args()
    run_rag(args)