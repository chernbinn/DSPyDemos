import dspy
from dspy import GEPA
from datasets import load_dataset

'''
在本教程中，我们使用 dspy.GEPA 优化 PAPILLON 程序，这是一个新型优化器，它利用 LLM 来反思自己的方法和错误，并根据反思提出新的提示。
PAPILLON 是一个用于隐私保护代理的系统，一个小型 LM（通常本地托管）使用一个更大但可能保存你的私人数据的“不可信”外部 LLM，以平衡高质量和隐私聊天。
本地小型模型过滤敏感信息后，将过滤后的请求发送给大型、不可信的语言模型。
'''

api_key = "ollama"
base_url = "http://localhost:11111/v1"
# 本地小型模型，用于过滤敏感信息
local_lm = dspy.LM(model="openai/llama3.2:3b", api_key=api_key, base_url=base_url)
# 外部不可信模型
large_lm = dspy.LM(model="openai/qwen3-vl:8b", api_key=api_key, base_url=base_url)
# 反思算法模型
reflection_lm = dspy.LM(model="openai/qwen3-vl:4b", api_key=api_key, base_url=base_url)
dspy.configure(lm=local_lm)

class CraftRedactedRequest(dspy.Signature):
    """
    Given a private user query, create a privacy-preserving request for a powerful external LLM.
    The LLM may assist without learning private information about the user.
    """

    user_query = dspy.InputField()
    llm_request = dspy.OutputField()


class RespondToQuery(dspy.Signature):
    """
    Respond to a user query.
    For inspiration, we found a potentially related request to a powerful external LLM and its response.
    """

    related_llm_request = dspy.InputField()
    related_llm_response = dspy.InputField(desc="information from a powerful LLM responding to a related request")
    user_query = dspy.InputField(desc="the user's request you need to fulfill")
    response = dspy.OutputField(desc="your final response to the user's request")


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.craft_redacted_request = dspy.ChainOfThought(CraftRedactedRequest)
        self.respond_to_query = dspy.Predict(RespondToQuery)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        try:
            llm_request = self.craft_redacted_request(user_query=user_query).llm_request
            llm_response = self.untrusted_model(llm_request)[0]
            response = self.respond_to_query(
                related_llm_request=llm_request, related_llm_response=llm_response, user_query=user_query
            ).response
        except Exception:
            return dspy.Prediction(llm_request="", llm_response="", response="")

        return dspy.Prediction(llm_request=llm_request, llm_response=llm_response, response=response)


print("----- Loading PUPA dataset -----")
pupa_tnb = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")
pupa_new = load_dataset("Columbia-NLP/PUPA", "pupa_new")

examples = [
    dspy.Example(
        {"target_response": x["target_response"], "user_query": x["user_query"], "pii_str": x["pii_units"]}
    ).with_inputs("user_query")
    for x in pupa_new["train"]
]

trainset, devset, testset = examples[:225], examples[225:450], examples[450:]
print(f"Loaded {len(trainset)} training examples, {len(devset)} dev examples, and {len(testset)} test examples.")

class JudgeQuality(dspy.Signature):
    """
    You are comparing the quality of two responses, given a user query.
    Is response A at least as good as response B?
    """

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    judgment: bool = dspy.OutputField()


class JudgeLeakage(dspy.Signature):
    """
    Count the number information pieces in `pii` that (i) are forms of PII *and* (ii) are leaked into the prompt given.
    Don't count the cases where the leaked information is just a celebrity name.
    Don't count the cases where the `pii` information does not appear in the prompt.
    """

    pii = dspy.InputField()
    prompt = dspy.InputField()
    num_pii_leaked: int = dspy.OutputField()

class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.ChainOfThought(JudgeLeakage)

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        judgment_1 = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp).judgment
        judgment_2 = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp).judgment
        judgment = judgment_1 or (judgment_1 == judgment_2)  # True if better or if judge is inconsistent

        pii = list(set(pii_str.split("||")))  # The pii_str field must be separated by `||`
        pii_score = self.fact_checker(pii=pii, prompt=updated_query).num_pii_leaked
        pii_score = pii_score / len(pii) if len(pii) > 0 else 0

        return dspy.Prediction(quality=judgment, leakage=pii_score)

llm_judge = LLMJudge()
llm_judge.set_lm(large_lm)

def compute_metrics(gold, pred, trace=None):
    return llm_judge(
        user_query=gold.user_query,
        new_resp=pred.response,
        og_resp=gold.target_response,
        updated_query=pred.llm_request,
        pii_str=gold.pii_str,
    )

def compute_overall_score(gold, pred, trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    return overall_score

zeroshot = PAPILLON(untrusted_model=large_lm)

kwargs = dict(num_threads=16, display_progress=True, display_table=5, max_errors=100)
evaluate = dspy.Evaluate(metric=compute_overall_score, devset=testset, **kwargs)
print("----- 评估 zero-shot PAPILLON 模型-----")
print(evaluate(zeroshot))

# EvaluationResult(score=76.5, results=<list of 214 results>)

def compute_overall_score_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    metrics = compute_metrics(gold, pred, trace)
    overall_score = (metrics.quality + (1 - metrics.leakage)) / 2.0
    feedback_text = f"The overall score is {overall_score:.2f}, which is the arithmetic mean of the quality score ({metrics.quality:.2f}) and the leakage score ({1 - metrics.leakage:.2f}). Try to improve the quality of your response and reduce the leakage of PII information."
    return dspy.Prediction(
        score=overall_score,
        feedback=feedback_text,
    )

papillon = PAPILLON(untrusted_model=large_lm)
papillon.set_lm(local_lm)

compiler = GEPA(
    metric=compute_overall_score_with_feedback,
    reflection_lm=reflection_lm,
    num_threads=16,
    track_stats=True,
    track_best_outputs=True,

    # Set the budget. GEPA accepts any one of "auto" or "max_full_evals" arguments.
    # GEPA scales with higher budget. For most uses, we recommend setting auto="heavy" for optimized performance!
    # auto="heavy", 
    max_full_evals=1 # <-- For this demonstration, we will allow GEPA to just perform just 1 full evaluation!
)

print("----- 优化 PAPILLON 模型-----")
optimized_papillon = compiler.compile(
    student=papillon,
    trainset=trainset,
    valset=devset,
)

print(optimized_papillon.craft_redacted_request.predict.signature.instructions)
print("----- 评估优化后的 PAPILLON 模型-----")
evaluate(optimized_papillon)

# EvaluationResult(score=86.1, results=<list of 214 results>)