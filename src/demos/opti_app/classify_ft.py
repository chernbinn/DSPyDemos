import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset
from dspy.clients.lm_local import LocalProvider

# pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
# pip install -U torch transformers==4.48.3 accelerate trl peft

print("----- 加载Banking77数据集 -----")                
# Load the Banking77 dataset.
CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
# CLASSES = load_dataset("PolyAI/banking77", split="train").features['label'].names
kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

# Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:1000]
]

random.Random(0).shuffle(raw_data)
print("分类数：", len(CLASSES), "部分分类：", CLASSES[:10])

unlabeled_trainset = [dspy.Example(text=x.text).with_inputs("text") for x in raw_data[:500]]
print("未标记训练集示例：", unlabeled_trainset[0])

classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")

student_lm_name = "llama3.2:3b"
student_lm = dspy.LM(model=f"openai/{student_lm_name}", 
        api_key="sk-",
        base_url="https://localhost:11111/v1",
        provider=LocalProvider(), 
        max_tokens=2000)
teacher_lm = dspy.LM(model='openai/qwen3-vl:8b', 
        api_key="sk-",
        base_url="https://localhost:11111/v1",
        max_tokens=3000)

student_classify = classify.deepcopy()
student_classify.set_lm(student_lm)

teacher_classify = classify.deepcopy()
teacher_classify.set_lm(teacher_lm)

'''
# Optional:
# [1] You can set `DSPY_FINETUNEDIR` environment variable to control where the directory that will be used to store the
#     checkpoints and fine-tuning data. If this is not set, `DSPY_CACHEDIR` is used by default.
# [2] You can set the `CUDA_VISIBLE_DEVICES` environment variable to control the GPU that will be used for fine-tuning
#     and inference. If this is not set and the default GPU that's used by HuggingFace's `transformers` library is
#     occupied, an OutOfMemoryError might be raised.
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["DSPY_FINETUNEDIR"] = "/path/to/dir"
'''
dspy.settings.experimental = True  # fine-tuning is an experimental feature, so we set a flag to enable it

print("----- 微调示例1 -----")
print("基于教师蒸馏方法，基于未标记训练集微调分类模型")
optimizer = dspy.BootstrapFinetune(num_threads=16)  # if you *do* have labels, pass metric=your_metric here!
classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=unlabeled_trainset)
classify_ft.get_lm().launch()
classify_ft(text="I didn't receive my money earlier and it says the transaction is still in progress. Can you fix it?")

devset = raw_data[500:600]
print("验证集示例：", devset[0])
metric = (lambda x, y, trace=None: x.label == y.label)
evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, display_table=5, num_threads=16)

print("----- 评估未标记数据优化模型效果 -----")
print("验证集评估结果：", evaluate(classify_ft))
classify_ft.get_lm().kill()

print("----- 微调示例2 -----")
print("基于教师蒸馏方法，基于有标签训练集微调分类模型")
optimizer = dspy.BootstrapFinetune(num_threads=16, metric=metric)
classify_ft = optimizer.compile(student_classify, teacher=teacher_classify, trainset=raw_data[:500])
classify_ft.get_lm().launch()

print("----- 评估有标签数据优化模型效果 -----")
print("验证集评估结果（微调模型）：", evaluate(classify_ft))
print("验证集评估结果（教师模型）：", evaluate(teacher_classify))

classify_ft(text="why hasnt my card come in yet?")
dspy.inspect_history()
