import asyncio, logging
from contextlib import AsyncExitStack

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(filename='/tmp/aterm2.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

SYSTEM = "You are an AI that combines direct help with friendly conversation. Share your knowledge openly, explain concepts thoroughly, and use tools when helpful. Focus on substance over process - engage naturally while delivering quality assistance."

async def ask(client, messages, tools, mcp_sessions_by_prefix, query):
    async def get_answer(client, messages, tools) -> dict:
        async with client.messages.stream(max_tokens=8192, messages=messages, model="claude-3-5-sonnet-latest",
                                          tools=tools, system=SYSTEM) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)

        final_message = await stream.get_final_message()

        return final_message.to_dict()

    async def process_answer(final_message, messages):
        messages.append({"role": "assistant", "content": final_message["content"]})
        for item in final_message["content"]:
            if item["type"] == "tool_use":
                prefix = item["name"].split("__")[0]
                session = mcp_sessions_by_prefix[prefix]
                result = await session.call_tool(item["name"].split("__")[-1], item["input"])
                result_text = "".join([x.text for x in result.content])
                if result.isError:
                    result_text = "ERROR: " + result_text
                print("\nresult:", result_text)
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": item["id"],
                        "content": result_text,
                    }],
                })

        return messages

    print(query)
    messages.append({"role": "user", "content": query})

    stop_reason = None
    i = 0
    while stop_reason in {None, "tool_use"} and i < 5:
        i += 1
        final_message = await get_answer(client, messages, tools)
        messages = await process_answer(final_message, messages)
        stop_reason = final_message["stop_reason"]
        print()

    return messages

async def init_mcp_session(prefix, mcp_config, exit_stack):
    read, write = await exit_stack.enter_async_context(stdio_client(StdioServerParameters(**mcp_config)))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await session.list_tools()
    session.tools = [{
        "name": prefix + x.name,
        "description": x.description,
        "input_schema": x.inputSchema
    } for x in tools.tools]

    return session

async def app(mcp_configs):
    client = AsyncAnthropic()
    async with AsyncExitStack() as exit_stack:
        init_tasks = [
            init_mcp_session(prefix + "__", mcp_config, exit_stack) for prefix, mcp_config in mcp_configs.items()
        ]
        mcp_sessions = await asyncio.gather(*init_tasks)
        tools, mcp_sessions_by_prefix = [], {}
        for prefix, session in zip(mcp_configs.keys(), mcp_sessions):
            mcp_sessions_by_prefix[prefix] = session
            for tool in session.tools:
                tools.append(tool)

        messages = []
        messages = await ask(
            client, messages, tools, mcp_sessions_by_prefix,
            "Please ping the echo tool with the message I'm a teapot and walk me through the process and explain the result ;) Make sure the message is echoed back to us."
        )

mcp_configs = {
    "everything1": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything"]
    },
}

if __name__ == "__main__":
    try:
        asyncio.run(app(mcp_configs), debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.exception("ERROR")
