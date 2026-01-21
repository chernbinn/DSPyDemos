import os
import dspy

'''
1.根据问题生成查询语句
2.根据查询语句在wiki中搜索相关内容
3.根据搜索结果回答问题
'''

class QueryGenerator(dspy.Signature):
    """Generate a query based on question to fetch relevant context"""
    question: str = dspy.InputField()
    query: str = dspy.OutputField()

def search_wikipedia(query: str) -> list[str]:
    """Query ColBERT endpoint, which is a knowledge source based on wikipedia data"""
    print(f"----------------search_wikipedia: {query}")
    results = dspy.ColBERTv2(url='https://www.zgbk.com/ecph/api/search')(query, k=1)
    print(f"ColBERTv2 query: {query}")
    print(results)
    print("-------------------")
    return [x["text"] for x in results]

class RAG(dspy.Module):
    def __init__(self):
        self.query_generator = dspy.Predict(QueryGenerator)
        self.answer_generator = dspy.ChainOfThought("question,context->answer")

    def forward(self, question, **kwargs):
        query = self.query_generator(question=question).query
        print(f"QueryGenerator query: {query}")
        context = search_wikipedia(query)[0]
        return self.answer_generator(question=question, context=context).answer

dspy.configure(lm=dspy.LM(
        model="ollama_chat/llama3.2:3b",
        base_url=f"http://192.168.3.17:11111",
        max_tokens=500,
        temperature=0.7))
rag = RAG()
print(rag(question="Is Lebron James the basketball GOAT?"))