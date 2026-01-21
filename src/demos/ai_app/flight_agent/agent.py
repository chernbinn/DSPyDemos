import dspy
from demos.ai_app.flight_agent.tools import *
from demos.ai_app.flight_agent.dummy_data import itinery_database

'''
订机票智能体demo
'''

def ollama():
    # 创建并配置Ollama模型实例
    # 方案1：可以正常调用llm，返回的内容解析存在问题，导致结果不符合预期
    # 属于自定义ollama llm，在实现中调用litellm，重写了部分dspy逻辑
    # ollama_lm = OllamaLMLiteLLM(model="qwen3:8b")
    # ollama_lm.configure(model_type="chat")
    # 方案2：可以正常调用llm，返回的内容解析符合预期
    # 完全使用dspy对litellm的调用
    # 模型配置：ollama_chat/llama3.2:3b，可以完成预定任务，无法修改订单
    # 模型配置：openai/llama3.2:3b，无法完成预定任务，存在返回内容格式错误
    ollama_lm = dspy.LM(
        model="ollama_chat/llama3.2:3b",
        api_key="ollama",
        base_url=f"http://192.168.3.17:11111/v1",
        max_tokens=500,
        temperature=0.7,
    )
    return ollama_lm

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent that helps user book and manage flights.

    You are given a list of tools to handle user request, and you should decide the right tool to use in order to
    fulfill users' request."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
                "Message that summarizes the process result, and the information users need, e.g., the "
                "confirmation_number if a new flight is booked."
            )
        )

agent = dspy.ReAct(
    DSPyAirlineCustomerService,
    tools = [
        fetch_flight_info,
        fetch_itinerary,
        pick_flight,
        book_flight,
        cancel_itinerary,
        get_user_info,
        file_ticket,
    ]
)

dspy.configure(lm=ollama())

# 第一部分：预订航班
print("=== 第一部分：预订航班 ===")
result = agent(user_request="please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam")
print(result)

print("\n------------------1 - 打印预订记录")
print(itinery_database)

# print("\n------------------2 - 打印llm调用记录")
# dspy.inspect_history(n=20)

print("\n------------------3 - 修改航班")
# 如果有预订记录，使用实际的确认号
if itinery_database:
    actual_confirmation_number = list(itinery_database.keys())[0]
    print(f"使用实际的确认号：{actual_confirmation_number}")
    result = agent(user_request=f"i want to take DA125 instead on 09/01, please help me modify my itinerary {actual_confirmation_number}")
    print(result)
else:
    print("没有找到预订记录，无法修改航班")


# 运行总结：
# 1. 方案1：可以正常调用llm，返回的内容解析存在问题，导致结果不符合预期
# 2. 方案2：可以正常调用llm，返回的内容解析符合预期
# 3. 对比两种方案，方案2的实现更符合dspy的设计，也更符合llm的调用规范
# 4. 方案2的实现中，重写了部分dspy逻辑，主要是在处理llm调用时，对返回内容的解析做了调整
# 5. 方案2最终结果符合预期，但是存在预定多张票的问题；且修改已有订单失败，从log看，代码解析回复格式存在问题
