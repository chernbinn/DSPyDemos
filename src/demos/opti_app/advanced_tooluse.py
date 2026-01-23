import dspy
import orjson
import random
from dspy.utils import download
import re, os
import inspect
from func_timeout import func_set_timeout

# pip install func_timeout datasets

gpt4o = dspy.LM("openai/llama3.2:3b", temperature=0.7)
dspy.configure(lm=gpt4o)

'''
本教程展示了DSPy优化器如何更有效地学习使用工具，从而提高工具调用场景中的准确性和效率。
'''

print("----- 准备数据 -----")
if not os.path.exists("ToolHop.json"):
    print("ToolHop.json not found, downloading...")
    download("https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/ToolHop.json")

data = orjson.loads(open("ToolHop.json").read())
random.Random(0).shuffle(data)

examples = []
fns2code = {}

def finish(answer: str):
    """Conclude the trajectory and return the final answer."""
    return answer

for datapoint in data:
    func_dict = {}
    for func_code in datapoint["functions"]:
        cleaned_code = func_code.rsplit("\n\n# Example usage", 1)[0]
        fn_name = re.search(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", cleaned_code)
        fn_name = fn_name.group(1) if fn_name else None

        if not fn_name:
            continue

        local_vars = {}
        # 把代码字符串转换为可执行的函数对象
        exec(cleaned_code, {}, local_vars)
        fn_obj = local_vars.get(fn_name)

        if callable(fn_obj):
            func_dict[fn_name] = fn_obj
            assert fn_obj not in fns2code, f"Duplicate function found: {fn_name}"
            fns2code[fn_obj] = (fn_name, cleaned_code)

    func_dict["finish"] = finish

    example = dspy.Example(question=datapoint["question"], answer=datapoint["answer"], functions=func_dict)
    examples.append(example.with_inputs("question", "functions"))

trainset, devset, testset = examples[:100], examples[100:400], examples[400:]

def wrap_function_with_timeout(fn):
    @func_set_timeout(10)
    def wrapper(*args, **kwargs):
        try:
            return {"return_value": fn(*args, **kwargs), "errors": None}
        except Exception as e:
            return {"return_value": None, "errors": str(e)}

    return wrapper

def fn_metadata(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or "No docstring."
    return dict(function_name=func.__name__, arguments=str(signature), docstring=docstring)

def metric(example, pred, trace=None):
    gold = str(example.answer).rstrip(".0").replace(",", "").lower()
    pred = str(pred.answer).rstrip(".0").replace(",", "").lower()
    return pred == gold  # stricter than the original paper's metric!

evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24, display_progress=True, display_table=0, max_errors=999)

class Agent(dspy.Module):
    def __init__(self, max_steps=5):
        self.max_steps = max_steps
        instructions = "For the final answer, produce short (not full sentence) answers in which you format dates as YYYY-MM-DD, names as Firstname Lastname, and numbers without leading 0s."
        signature = dspy.Signature('question, trajectory, functions -> next_selected_fn, args: dict[str, Any]', instructions)
        self.react = dspy.ChainOfThought(signature)

    def forward(self, question, functions):
        tools = {fn_name: fn_metadata(fn) for fn_name, fn in functions.items()}
        trajectory = []

        for _ in range(self.max_steps):
            pred = self.react(question=question, trajectory=trajectory, functions=tools)
            selected_fn = pred.next_selected_fn.strip('"').strip("'")
            fn_output = wrap_function_with_timeout(functions[selected_fn])(**pred.args)
            trajectory.append(dict(reasoning=pred.reasoning, selected_fn=selected_fn, args=pred.args, **fn_output))

            if selected_fn == "finish":
                break

        return dspy.Prediction(answer=fn_output.get("return_value", ''), trajectory=trajectory)

agent = Agent()
print("----- 评估未优化Agent -----")
print(evaluate(agent))

print("----- 优化Agent -----")
simba = dspy.SIMBA(metric=metric, max_steps=12, max_demos=10)
optimized_agent = simba.compile(agent, trainset=trainset, seed=6793115)

print("----- 评估优化后的Agent -----")
print(evaluate(optimized_agent))
