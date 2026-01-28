
'''
在本指南中，我们将向您介绍如何在您的DSPy程序中启用流式处理。DSPy流式处理包含两个部分：
1.输出令牌流式处理：在生成时流式传输单个令牌，而不是等待完整响应。
2.中间状态流式处理：提供有关程序执行状态的真实时更新（例如，“正在调用网络搜索...”，“正在处理结果...”）
'''

'''
token流式输出：
DSPy的token流式传输功能适用于您管道中的任何模块，而不仅仅是最终输出。唯一的要求是流式传输的字段必须是str类型。要启用token流式传输：
1.用dspy.streamify包装您的程序
2.创建一个或多个dspy.streaming.StreamListener对象来指定要流式传输的字段
'''
import os
import asyncio
import dspy

os.environ["OPENAI_API_KEY"] = "ollama"
lm=dspy.LM(model="openai/llama3.2:3b",
           api_key="sk-1234567890abcdef1234567890abcdef",
           api_base="https://localhost:11111/v1",
           )

def token_streaming():
    '''
    单个token流式输出，由于缓存的原因，最后一个流式回复包含多个token。当所有token都输出完成后，返回一个完成的prediction对象。
    '''
    dspy.configure(lm=lm)

    predict = dspy.Predict("question->answer")

    # Enable streaming for the 'answer' field
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )

    async def read_output_stream():
        output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

        async for chunk in output_stream:
            print(chunk)

    asyncio.run(read_output_stream())

    '''
    StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
    StreamResponse(predict_name='self', signature_field_name='answer', chunk=' get')
    StreamResponse(predict_name='self', signature_field_name='answer', chunk=' to')
    StreamResponse(predict_name='self', signature_field_name='answer', chunk=' the')
    StreamResponse(predict_name='self', signature_field_name='answer', chunk=' other')
    StreamResponse(predict_name='self', signature_field_name='answer', chunk=' side of the frying pan!')
    Prediction(
        answer='To get to the other side of the frying pan!'
    )
    说明：
    predict_name：包含 signature_field_name 的预测名称。该名称与运行 your_program.named_predictors() 时使用的键名相同。
    在上述代码中，因为 answer 来自 predict 本身，所以 predict_name 显示为 self ，这是运行 predict.named_predictors() 时唯一的键。
    signature_field_name：这些标记映射到的输出字段。 predict_name 和 signature_field_name 一起构成字段的唯一标识符。
    我们将在本指南的后面演示如何处理多个字段的流式传输和重复的字段名。
    chunk：流式数据块的值。

    当找到缓存结果时，流将跳过单个标记，并且只生成最终的 Prediction 对象。
    '''

class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer

def streaming_multiple_fields():
    '''
    多个字段流式输出：
    您可以为多个字段启用流式传输。例如，假设您的程序有一个签名 question -> answer, feedback ，
    您可以为 answer 和 feedback 字段启用流式传输。
    '''
    dspy.configure(lm=lm)
    predict = MyModule()
    stream_listeners = [
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
    ]
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=stream_listeners,
    )

    async def read_output_stream():
        output = stream_predict(question="why did a chicken cross the kitchen?")

        return_value = None
        async for chunk in output:
            if isinstance(chunk, dspy.streaming.StreamResponse):
                print(chunk)
            elif isinstance(chunk, dspy.Prediction):
                return_value = chunk
        return return_value

    program_output = asyncio.run(read_output_stream())
    print("Final output: ", program_output)

'''
默认情况下，一个StreamListener在完成一次流式会话后会自动关闭。这种设计有助于防止性能问题，
因为每个token都会广播给所有配置的流式监听器，而太多的活动监听器可能会引入显著的开销。
然而，在重复循环中使用DSPy模块的情况下——例如使用dspy.ReAct——您可能希望每次使用时都从每个
预测中流式传输相同的字段。要启用此行为，在创建StreamListener时将allow_reuse设置为True。
'''
dspy.configure(lm=lm)
def fetch_user_info(user_name: str):
    """Get user information like name, birthday, etc."""
    return {
        "name": user_name,
        "birthday": "2009-05-16",
    }

def get_sports_news(year: int):
    """Get sports news for a given year."""
    if year == 2009:
        return "Usane Bolt broke the world record in the 100m race."
    return None

react = dspy.ReAct("question->answer", tools=[fetch_user_info, get_sports_news])
stream_listeners = [
    # dspy.ReAct has a built-in output field called "next_thought".
    dspy.streaming.StreamListener(signature_field_name="next_thought", allow_reuse=True),
]
stream_react = dspy.streamify(react, stream_listeners=stream_listeners)

async def read_output_stream():
    output = stream_react(question="What sports news happened in the year Adam was born?")
    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value

def stream_reuse():
    print(asyncio.run(read_output_stream()))

def stream_duplicate_field():
    '''
    重复字段流式输出：
    假设您的程序有两个签名：question->answer ，question, answer->answer, score
    您可以为分别为同名称的 answer 字段启用流式传输。
    '''
    dspy.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()

            self.predict1 = dspy.Predict("question->answer")
            self.predict2 = dspy.Predict("question, answer->answer, score")

        def forward(self, question: str, **kwargs):
            answer = self.predict1(question=question)
            simplified_answer = self.predict2(answer=answer)
            return simplified_answer

    predict = MyModule()
    stream_listeners = [
        dspy.streaming.StreamListener(
            signature_field_name="answer",
            predict=predict.predict1,
            predict_name="predict1"
        ),
        dspy.streaming.StreamListener(
            signature_field_name="answer",
            predict=predict.predict2,
            predict_name="predict2"
        ),
    ]
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=stream_listeners,
    )

    async def read_output_stream():
        output = stream_predict(question="why did a chicken cross the kitchen?")

        return_value = None
        async for chunk in output:
            if isinstance(chunk, dspy.streaming.StreamResponse):
                print(chunk)
            elif isinstance(chunk, dspy.Prediction):
                return_value = chunk
        return return_value

    program_output = asyncio.run(read_output_stream())
    print("Final output: ", program_output)

'''
状态流式传输让用户了解程序的进展情况，特别适用于长时间运行的操作，如工具调用或复杂的 AI 流程。要实现状态流式传输：
1.通过子类化 dspy.streaming.StatusMessageProvider 创建一个自定义状态消息提供者
2.覆盖所需钩子方法以提供自定义状态消息
3.将您的提供者传递给 dspy.streamify
示例：
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def lm_end_status_message(self, outputs):
        return f"Tool finished with output: {outputs}!"
'''
def middle_status_stream():
    dspy.configure(lm=lm)
    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()

            self.tool = dspy.Tool(lambda x: 2 * x, name="double_the_number")
            self.predict = dspy.ChainOfThought("num1, num2->sum")

        def forward(self, num, **kwargs):
            num2 = self.tool(x=num)
            return self.predict(num1=num, num2=num2)

    class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
        def tool_start_status_message(self, instance, inputs):
            return f"Calling Tool {instance.name} with inputs {inputs}..."

        def tool_end_status_message(self, outputs):
            return f"Tool finished with output: {outputs}!"

    predict = MyModule()
    stream_listeners = [
        # dspy.ChainOfThought has a built-in output field called "reasoning".
        dspy.streaming.StreamListener(signature_field_name="reasoning"),
    ]
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=stream_listeners,
        status_message_provider=MyStatusMessageProvider(),
    )

    async def read_output_stream():
        output = stream_predict(num=3)

        return_value = None
        async for chunk in output:
            if isinstance(chunk, dspy.streaming.StreamResponse):
                print(chunk)
            elif isinstance(chunk, dspy.Prediction):
                return_value = chunk
            elif isinstance(chunk, dspy.streaming.StatusMessage):
                print(chunk)
        return return_value


    program_output = asyncio.run(read_output_stream())
    print("Final output: ", program_output)

'''
默认情况下，调用流化 DSPy 程序会产生一个异步生成器。若要获取同步生成器，可以设置标志 async_streaming=False
'''
def sync_stream():
    dspy.configure(lm=lm)
    predict = dspy.Predict("question->answer")

    # Enable streaming for the 'answer' field
    stream_predict = dspy.streamify(
        predict,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        async_streaming=False,
    )

    output = stream_predict(question="why did a chicken cross the kitchen?")

    program_output = None
    for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            program_output = chunk
    print(f"Program output: {program_output}")

if __name__ == "__main__":  
    token_streaming()
    streaming_multiple_fields()
    stream_reuse()
    stream_duplicate_field()
    middle_status_stream()
    sync_stream()
