
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "ollama"
lm=dspy.LM(model="openai/llama3.2:3b",
           api_key="sk-1234567890abcdef1234567890abcdef",
           api_base="https://localhost:11111/v1",
           )

def async_call():
    dspy.configure(lm=lm)
    predict = dspy.Predict("question->answer")

    async def main():
        # Use acall() for async execution
        output = await predict.acall(question="why did a chicken cross the kitchen?")
        print(output)


    asyncio.run(main())

def async_use_tool():
    async def foo(x):
        # Simulate an async operation
        await asyncio.sleep(0.1)
        print(f"I get: {x}")

    # Create a tool from the async function
    tool = dspy.Tool(foo)

    async def main():
        # Execute the tool asynchronously
        await tool.acall(x=2)

    asyncio.run(main())

def async_tool_in_sync_context():
    async def async_tool(x: int) -> int:
        """An async tool that doubles a number."""
        await asyncio.sleep(0.1)
        return x * 2

    tool = dspy.Tool(async_tool)

    # Option 1: Use context manager for temporary conversion
    with dspy.context(allow_tool_async_sync_conversion=True):
        result = tool(x=5)  # Works in sync context
        print(result)  # 10

    # Option 2: Configure globally
    dspy.configure(allow_tool_async_sync_conversion=True)
    result = tool(x=5)  # Now works everywhere
    print(result)  # 10

def async_module():
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MyModule(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.ChainOfThought("question->answer")
            self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

        async def aforward(self, question, **kwargs):
            # Execute predictions sequentially but asynchronously
            answer = await self.predict1.acall(question=question)
            return await self.predict2.acall(answer=answer)


    async def main():
        mod = MyModule()
        result = await mod.acall(question="Why did a chicken cross the kitchen?")
        print(result)

asyncio.run(main())

if __name__ == "__main__":
    async_call()
    async_use_tool()
    async_tool_in_sync_context()
    async_module()

