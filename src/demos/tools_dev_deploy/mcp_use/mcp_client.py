
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

# Create server parameters for stdio connection
current_dir = os.path.dirname(os.path.abspath(__file__))
server_params = StdioServerParameters(
    command="python",  # Executable
    args=[os.path.join(current_dir, "mcp_server.py")],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())