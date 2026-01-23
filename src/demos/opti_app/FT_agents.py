#> pip install -U alfworld==0.3.5 multiprocess
#> alfworld-download

import dspy
from dspy.datasets.alfworld import AlfWorld

'''
通过微调学习优化基于代理的复杂系统。本教程演示了如何在像游戏这样的交互环境中提高代理性能。

> pip install -U alfworld==0.3.5 multiprocess
> alfworld-download
'''

base_url = "https://localhost:11111/v1"
api_key = "sk-1234567890abcdef1234567890abcdef"

gpt4o_mini = dspy.LM('openai/qwen3-vl:8b', base_url=base_url, api_key=api_key)
gpt4o = dspy.LM('openai/qwen3-vl:4b', base_url=base_url, api_key=api_key)
# experimental=True 是 DSPy 中启用实验性功能的配置选项
dspy.configure(experimental=True)

alfworld = AlfWorld()
trainset, devset = alfworld.trainset[:200], alfworld.devset[-200:]
print(f"Trainset size: {len(trainset)}, Devset size: {len(devset)}")

example = trainset[0]

with alfworld.POOL.session() as env:
    task, info = env.init(**example.inputs())

print("Example task:")
print(task)

class Agent(dspy.Module):
    def __init__(self, max_iters=50, verbose=False):
        self.max_iters = max_iters
        self.verbose = verbose
        self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

    def forward(self, idx):
        with alfworld.POOL.session() as env:
            trajectory = []
            task, info = env.init(idx)
            if self.verbose:
                print(f"Task: {task}")

            for _ in range(self.max_iters):
                trajectory_ = "\n".join(trajectory)
                possible_actions = info["admissible_commands"][0] + ["think: ${...thoughts...}"]
                prediction = self.react(task=task, trajectory=trajectory_, possible_actions=possible_actions)
                trajectory.append(f"> {prediction.action}")

                if prediction.action.startswith("think:"):
                    trajectory.append("OK.")
                    continue

                obs, reward, done, info = env.step(prediction.action)
                obs, reward, done = obs[0], reward[0], done[0]
                trajectory.append(obs)

                if self.verbose:
                    print("\n".join(trajectory[-2:]))

                if done:
                    break

        assert reward == int(info["won"][0]), (reward, info["won"][0])
        return dspy.Prediction(trajectory=trajectory, success=reward)

'''
simple:
self.react = dspy.Predict("task, trajectory, possible_actions: list[str] -> action")

optimization:
INSTRUCTIONS = """
Interact with a simulated household to achieve a high-level goal. Make sure to plan, track subgoals,
determine likely locations for common household items (e.g. desklamps will likely be on desks, shelfs, or dressers),
and explore systematically (e.g. check all desks one by one for desklamp).
""".strip()

self.react = dspy.Predict(dspy.Signature("task, trajectory, possible_actions: list[str] -> action", INSTRUCTIONS))
'''
agent_4o = Agent()
agent_4o.set_lm(gpt4o)
agent_4o.verbose = True

print("----- 调用一次模型 -----")
agent_4o(**example.inputs())

metric = lambda x, y, trace=None: y.success
evaluate = dspy.Evaluate(devset=devset, metric=metric, display_progress=True, num_threads=16)

agent_4o.verbose = False
print("----- 评估大模型 -----")
evaluate(agent_4o)

agent_4o_mini = Agent()
agent_4o_mini.set_lm(gpt4o_mini)
print("----- 评估小模型 -----")
evaluate(agent_4o_mini)

optimizer = dspy.MIPROv2(metric=metric, auto="light", num_threads=16, prompt_model=gpt4o)

config = dict(max_bootstrapped_demos=1, max_labeled_demos=0, minibatch_size=40)
print("----- 编译优化大模型 -----")
optimized_4o = optimizer.compile(agent_4o, trainset=trainset, **config)

'''
student初始状态对比：
方法	        学生模型初始状态	        优点	                缺点
deepcopy()	拥有 teacher 的结构和参数	 1. 继承 teacher 的知识     需要复制操作
                                      2. 更好的初始化
                                      3. 更快收敛	
从头开始	 随机初始化	                 1. 完全独立
                                      2. 可能发现新解	          1. 收敛慢
                                                                2. 可能学不到知识
'''
# 学生初始状态数据拷贝自老师。
student_4o_mini = optimized_4o.deepcopy()
student_4o_mini.set_lm(gpt4o_mini)
# student_4o_mini.react.demos = []  # you can optionally reset the demos

print("----- 编译微调小模型:蒸馏模式 -----")
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=16)
finetuned_4o_mini = optimizer.compile(student_4o_mini, teacher=optimized_4o, trainset=trainset)
print("----- 评估微调后的学生模型 -----")
evaluate(finetuned_4o_mini)

finetuned_4o_mini.save('finetuned_4o_mini_001.pkl')

finetuned_4o_mini.verbose = True
print("----- 调用微调后的学生模型 -----")
finetuned_4o_mini(**devset[0].inputs())

'''
loaded = Agent()
loaded.load('finetuned_4o_mini_001.pkl', allow_pickle=True)
'''