

# pip install datasets soundfile torch==2.0.1+cu118 torchaudio==2.0.2+cu118

import random
import dspy
from dspy.datasets import DataLoader
from collections import defaultdict
import os
import base64
import hashlib
from openai import OpenAI
import torch
import torchaudio
import soundfile as sf
import io
from IPython.display import Audio

'''
示例1：基于音频信息回答问题
示例2：基于文本信息及声音风格（情绪，如愤怒、开心等）要求输出相应的音频

依赖多模态能处理音频数据的模型，如gpt-4o-mini-audio-preview-2024-12-17
'''

def demo1():
    print("------------ 示例1：基于音频信息回答问题")
    print("-----加载SpokenSquad数据集-----")

    dspy.configure(lm=dspy.LM(model='gpt-4o-mini-audio-preview-2024-12-17'))

    kwargs = dict(fields=("context", "instruction", "answer"), input_keys=("context", "instruction"))
    spoken_squad = DataLoader().from_huggingface(dataset_name="AudioLLMs/spoken_squad_test", split="train", trust_remote_code=True, **kwargs)

    random.Random(42).shuffle(spoken_squad)
    spoken_squad = spoken_squad[:100]

    split_idx = len(spoken_squad) // 2
    trainset_raw, testset_raw = spoken_squad[:split_idx], spoken_squad[split_idx:]

    def preprocess(x):
        audio = dspy.Audio.from_array(x.context["array"], x.context["sampling_rate"])
        return dspy.Example(
            passage_audio=audio,
            question=x.instruction,
            answer=x.answer
        ).with_inputs("passage_audio", "question")

    trainset = [preprocess(x) for x in trainset_raw]
    testset = [preprocess(x) for x in testset_raw]

    print(f"trainset size: {len(trainset)}, testset size: {len(testset)}")

    class SpokenQASignature(dspy.Signature):
        """Answer the question based on the audio clip."""
        passage_audio: dspy.Audio = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc = 'factoid answer between 1 and 5 words')

    spoken_qa = dspy.ChainOfThought(SpokenQASignature)
    evaluate_program = dspy.Evaluate(devset=testset, metric=dspy.evaluate.answer_exact_match,display_progress=True, num_threads = 10, display_table=True)

    print("-----评估未优化模型-----")
    evaluate_program(spoken_qa)

    optimizer = dspy.BootstrapFewShotWithRandomSearch(metric = dspy.evaluate.answer_exact_match, max_bootstrapped_demos=2, max_labeled_demos=2, num_candidate_programs=5)
    print("-----执行模型优化-----")
    optimized_program = optimizer.compile(spoken_qa, trainset = trainset)
    print("-----评估优化后的模型-----")
    evaluate_program(optimized_program)

    print("--------构建教师优化器---------")
    prompt_lm = dspy.LM(model='gpt-4o-mini') #NOTE - this is the LLM guiding the MIPROv2 instruction candidate proposal
    optimizer = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", prompt_model = prompt_lm)
    print("-----执行教师优化-----")
    #NOTE - MIPROv2's dataset summarizer cannot process the audio files in the dataset, so we turn off the data_aware_proposer 
    optimized_program = optimizer.compile(spoken_qa, trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2, data_aware_proposer=False)
    print("-----评估教师优化后的模型-----")
    evaluate_program(optimized_program)

# --------------------------------------------
CACHE_DIR = ".audio_cache"
def hash_key(raw_line: str, prompt: str) -> str:
        return hashlib.sha256(f"{raw_line}|||{prompt}".encode("utf-8")).hexdigest()

def generate_dspy_audio(raw_line: str, prompt: str) -> dspy.Audio:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    key = hash_key(raw_line, prompt)
    wav_path = os.path.join(CACHE_DIR, f"{key}.wav")
    if not os.path.exists(wav_path):
        print(f"Generating audio for {raw_line} with prompt {prompt}")
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral", #NOTE - this can be configured to any of the 11 offered OpenAI TTS voices - https://platform.openai.com/docs/guides/text-to-speech#voice-options. 
            input=raw_line,
            instructions=prompt,
            response_format="wav"
        )
        with open(wav_path, "wb") as f:
            f.write(response.content)
    with open(wav_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return dspy.Audio(data=encoded, format="wav")

class EmotionStylePromptSignature(dspy.Signature):
    """Generate an OpenAI TTS instruction that makes the TTS model speak the given line with the target emotion or style."""
    raw_line: str = dspy.InputField()
    target_style: str = dspy.InputField()
    openai_instruction: str = dspy.OutputField()

class EmotionStylePrompter(dspy.Module):
    def __init__(self):
        self.prompter = dspy.ChainOfThought(EmotionStylePromptSignature)

    def forward(self, raw_line, target_style):
        # 模型基于内容和情感生成OpenAI TTS指令
        out = self.prompter(raw_line=raw_line, target_style=target_style)
        # 模型基于内容与tts指令生成音频
        audio = generate_dspy_audio(raw_line, out.openai_instruction)
        return dspy.Prediction(audio=audio)

def demo2():
    # 用于生成不同情感语音的语音指令模型
    dspy.configure(lm=dspy.LM(model='gpt-4o-mini'))
    
    print("------------ 示例2：基于文本内容生成不同情感的语音")
    print("-----加载Crema-D数据集-----")
    label_map = ['neutral', 'happy', 'sad', 'anger', 'fear', 'disgust']
    kwargs = dict(fields=("sentence", "label", "audio"), input_keys=("sentence", "label"))
    crema_d = DataLoader().from_huggingface(dataset_name="myleslinder/crema-d", split="train", trust_remote_code=True, **kwargs)

    def preprocess(x):
        return dspy.Example(
            raw_line=x.sentence,
            target_style=label_map[x.label],
            reference_audio=dspy.Audio.from_array(x.audio["array"], x.audio["sampling_rate"])
        ).with_inputs("raw_line", "target_style")

    random.Random(42).shuffle(crema_d)
    crema_d = crema_d[:100]

    random.seed(42)
    label_to_indices = defaultdict(list)
    for idx, x in enumerate(crema_d):
        label_to_indices[x.label].append(idx)

    per_label = 100 // len(label_map)
    train_indices, test_indices = [], []
    for indices in label_to_indices.values():
        selected = random.sample(indices, min(per_label, len(indices)))
        split = len(selected) // 2
        train_indices.extend(selected[:split])
        test_indices.extend(selected[split:])

    trainset = [preprocess(crema_d[idx]) for idx in train_indices]
    testset = [preprocess(crema_d[idx]) for idx in test_indices]

    os.makedirs(CACHE_DIR, exist_ok=True) 
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().eval()

    def decode_dspy_audio(dspy_audio):
        audio_bytes = base64.b64decode(dspy_audio.data)
        array, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return torch.tensor(array).unsqueeze(0)

    def extract_embedding(audio_tensor):
        with torch.inference_mode():
            return model(audio_tensor)[0].mean(dim=1)

    def cosine_similarity(a, b):
        return torch.nn.functional.cosine_similarity(a, b).item()

    def audio_similarity_metric(example, pred, trace=None):
        ref_audio = decode_dspy_audio(example.reference_audio)
        gen_audio = decode_dspy_audio(pred.audio)

        ref_embed = extract_embedding(ref_audio)
        gen_embed = extract_embedding(gen_audio)

        score = cosine_similarity(ref_embed, gen_embed)

        if trace is not None:
            return score > 0.8 
        return score

    evaluate_program = dspy.Evaluate(devset=testset, metric=audio_similarity_metric, display_progress=True, num_threads = 10, display_table=True)

    print("-----评估未优化模型-----")
    evaluate_program(EmotionStylePrompter())

    program = EmotionStylePrompter()
    print("-----调用模型-----")
    pred = program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)
    print(audio_similarity_metric(testset[1], pred)) #0.5725605487823486
    dspy.inspect_history(n=1)

    audio_bytes = base64.b64decode(pred.audio.data)
    array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    # 输出语音播放控件
    Audio(array, rate=rate)

    print("-----构建教师优化器-----")
    prompt_lm = dspy.LM(model='gpt-4o-mini')
    teleprompter = dspy.MIPROv2(metric=audio_similarity_metric, auto="light", prompt_model = prompt_lm)
    print("-----执行教师优化-----")
    optimized_program = teleprompter.compile(EmotionStylePrompter(),trainset=trainset)
    print("-----评估教师优化后的模型-----")
    evaluate_program(optimized_program)
    print("-----调用教师优化后的模型-----")
    pred = optimized_program(raw_line=testset[1].raw_line, target_style=testset[1].target_style)

    print(audio_similarity_metric(testset[1], pred)) #0.6691027879714966
    dspy.inspect_history(n=1)

    # 显示教师优化后的模型生成的语音控件
    audio_bytes = base64.b64decode(pred.audio.data)
    array, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    Audio(array, rate=rate)