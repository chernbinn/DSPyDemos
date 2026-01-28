
'''
BestOfN 和 Refine 都是 DSPy 模块，旨在通过使用不同的 rollout ID 进行多次 LM 调用来提高预测的可靠性和质量，从而绕过缓存。
这两个模块会在达到 N 次尝试或 reward_fn 返回的奖励超过 threshold 时停止。

BestOfN 是一个模块，它使用不同的 rollout ID 多次（最多 N 次）运行提供的模块。它返回第一个通过指定阈值的预测结果，
或者如果没有结果达到阈值，则返回奖励最高的那个。
'''

import dspy

dspy.configure(lm=dspy.LM(
    model="ollama_chat/llama3.2:3b",
    base_url="http://localhost:11111",
    api_key="ollama",
    ))

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

def normal_bestof():
    print("------- normal bestof -------")
    '''
    从模型中获得一个单词的答案。我们可以使用 BestOfN 尝试多个 rollout ID 并返回最佳结果
    '''
    best_of_3 = dspy.BestOfN(
        module=dspy.ChainOfThought("question -> answer"), 
        N=3, 
        reward_fn=one_word_answer, 
        threshold=1.0
    )

    result = best_of_3(question="What is the capital of Belgium?")
    print(result.answer)  # Brussels

def error_bestof():
    print("------- error bestof -------")
    '''
    默认情况下，如果模块在尝试过程中遇到错误，它将继续尝试直到达到 N 次尝试。您可以通过 fail_count 参数调整此行为
    '''
    best_of_3 = dspy.BestOfN(
        module=qa,  # qa不初始化，模拟一个错误
        N=3, 
        reward_fn=one_word_answer,
        threshold=1.0,
        fail_count=1
    )

    best_of_3(question="What is the capital of Belgium?")
    # raises an error after the first failure


'''
Refine通过添加自动反馈循环扩展了BestOfN的功能。在每次不成功的尝试（除最后一次外），
它会自动生成关于模块性能的详细反馈，并使用这些反馈作为后续运行的提示。
'''
def normal_refine():
    print("------- normal refine -------")
    refine = dspy.Refine(
        module=dspy.ChainOfThought("question -> answer"), 
        N=3, 
        reward_fn=one_word_answer, 
        threshold=1.0
    )

    result = refine(question="What is the capital of Belgium?")
    print(result.answer)  # Brussels

def error_refine():
    print("------- error refine -------")
    # Stop after the first error
    refine = dspy.Refine(
        module=qa, 
        N=3, 
        reward_fn=one_word_answer, 
        threshold=1.0,
        fail_count=1
    )

# ----------------------------------------------
class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)
def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Reward the summary for being close to 75 words with a tapering off for longer summaries.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

def bestof_vs_refine():
    '''
    这两个模块具有相似的功能，但它们的方法不同：
    1.BestOfN只是尝试不同的rollout ID，并根据reward_fn定义的奖励选择最佳结果预测。
    2.Refine增加了一个反馈循环，使用lm根据先前的预测和reward_fn中的代码生成关于模块自身性能的详细反馈。
    然后，这个反馈被用作后续运行的提示。
    '''
    print("----- bestof vs refine -----")
    print("-------------- refined")
    refined_qa = dspy.Refine(
        module=dspy.ChainOfThought("question -> answer"),
        N=3,
        reward_fn=factuality_reward,
        threshold=1.0
    )

    result = refined_qa(question="Tell me about Belgium's capital city.")
    print(result.answer)

    print("-------------- bestof")
    optimized_summarizer = dspy.BestOfN(
        module=dspy.ChainOfThought("text -> summary"),
        N=50,
        reward_fn=ideal_length_reward,
        threshold=0.9
    )

    result = optimized_summarizer(
        text="[Long text to summarize...]"
    )
    print(result.summary)

if __name__ == "__main__":
    normal_bestof()
    error_bestof()
    input("Press Enter to continue...")
    normal_refine()
    error_refine()
    input("Press Enter to continue...")
    bestof_vs_refine()


