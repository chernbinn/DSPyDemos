# ProgramOfThought demos

import dspy

lm = dspy.LM("ollama/llama3.1:8b", api_base="http://localhost:11111", api_key="None")
dspy.configure(lm=lm)

class BasicGenerateAnswer(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

def pot_demo():
    print("-----ProgramOfThought demo-----")
    pot = dspy.ProgramOfThought(BasicGenerateAnswer)
    problem = "2*5 + 4"
    pot(question=problem).answer
    print("problem:", problem)
    print("pot answer:", pot(question=problem).answer)
    print("-----inspect history-----")
    dspy.inspect_history()
    print()

def cot_pot_compare_demo():
    print("-----ChainOfThought vs ProgramOfThought demo-----")
    problem = "Compute 12! / sum of prime numbers between 1 and 30."

    cot = dspy.ChainOfThought(BasicGenerateAnswer)
    cot(question=problem).answer
    print("problem:", problem)
    print("cot answer:", cot(question=problem).answer)
    print("-----inspect history-----")
    dspy.inspect_history()
    print()
    pot = dspy.ProgramOfThought(BasicGenerateAnswer)
    print("pot answer:", pot(question=problem).answer)
    print("-----inspect history-----")
    dspy.inspect_history()
    print()

def search_wikipedia(query: str):
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer the non-numerical components of a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

from dspy.dsp.utils import deduplicate

class MultiHopSearchWithPoT(dspy.Module):
    def __init__(self, num_hops):
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ProgramOfThought(GenerateAnswer, max_iters=3)

    def forward(self, question):
        context = []
        for _ in range(self.num_hops):
            query = self.generate_query(context=context, question=question).query
            context = deduplicate(context + search_wikipedia(query))
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

def complex_pot_demo():
    multi_hop_pot = MultiHopSearchWithPoT(num_hops=2)
    question = (
        "What is the square of the total sum of the atomic number of the metal "
        "that makes up the gift from France to the United States in the late "
        "19th century and the sum of the number of digits in the first 10 prime numbers?"
    )
    multi_hop_pot(question=question).answer
    dspy.inspect_history()

if __name__ == "__main__":
    pot_demo()
    cot_pot_compare_demo()
    print("Next demo depends on wikipedia search, please make sure you have connected to the internet.")
    continue_ = input("Do you want to continue? (y/n)")
    if continue_ == "y":
        complex_pot_demo()



