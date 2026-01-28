import dspy
import os

'''
通过本教程，您将了解如何使用 MLflow Tracing 调试问题并提升可观测性。您还将探索如何使用回调构建自定义日志解决方案。
MLflow相对复杂，不在该demo中示例。
'''
'''
DSPy 提供了 inspect_history() 工具，它会打印出 LLM 调用
'''
os.environ["OPENAI_API_KEY"] = "sk-1234567890abcdef1234567890abcdef"
lm = dspy.LM(model="openai/llama3.2:3b",
           api_base="https://localhost:11111/v1",
           )

def simple_debug():
    '''
    本函数演示了如何使用 DSPy 调试工具 inspect_history() 来打印 LLM 调用。
    '''
    colbert = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    dspy.configure(lm=lm)


    def retrieve(query: str):
        """Retrieve top 3 relevant information from ColBert"""
        results = colbert(query, k=3)
        return [x["text"] for x in results]


    agent = dspy.ReAct("question -> answer", tools=[retrieve], max_iters=3)

    prediction = agent(question="Which baseball team does Shohei Ohtani play for in June 2025?")
    print(prediction.answer)

    # Print out 5 LLM calls
    dspy.inspect_history(n=5)

'''
有时，您可能需要实现自定义的日志解决方案。例如，您可能需要记录由特定模块触发的事件。
DSPy 的回调机制支持此类用例。 BaseCallback 类提供了几个用于自定义日志行为的处理器

在回调中处理输入或输出数据时要小心。就地修改它们可能会修改传递给程序的原始数据，从而
导致意外行为。为了避免这种情况，强烈建议在执行任何可能修改数据的操作之前创建数据的副本。
'''

from dspy.utils.callback import BaseCallback

def custom_logging():
    '''
    本函数演示了如何使用 DSPy 回调机制实现自定义日志解决方案。
    '''
    # 1. Define a custom callback class that extends BaseCallback class
    class AgentLoggingCallback(BaseCallback):

        # 2. Implement on_module_end handler to run a custom logging code.
        def on_module_end(self, call_id, outputs, exception):
            step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
            print(f"== {step} Step ===")
            for k, v in outputs.items():
                print(f"  {k}: {v}")
            print("\n")

        def _is_reasoning_output(self, outputs):
            return any(k.startswith("Thought") for k in outputs.keys())

    # 3. Set the callback to DSPy setting so it will be applied to program execution
    dspy.configure(callbacks=[AgentLoggingCallback()])

if __name__ == "__main__":
    simple_debug()
    custom_logging()