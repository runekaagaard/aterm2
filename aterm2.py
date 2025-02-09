import asyncio, logging, json, argparse, os
from contextlib import AsyncExitStack

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

logging.basicConfig(filename='/tmp/aterm2.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

SYSTEM = "You are an AI that combines direct help with friendly conversation. Share your knowledge openly, explain concepts thoroughly, and use tools when helpful. Focus on substance over process - engage naturally while delivering quality assistance."

def printer(printer_type, text):
    if printer_type == "llm_text_stream":
        print(text, end="", flush=True)
    else:
        print(f"{printer_type}:", text, flush=True)

async def claude(client, messages, tools):
    async with client.messages.stream(max_tokens=8192, messages=messages, model="claude-3-5-sonnet-latest",
                                      tools=tools, system=SYSTEM) as stream:
        async for text in stream.text_stream:
            printer("llm_text_stream", text)

    final_message = await stream.get_final_message()
    return final_message.to_dict()

async def tools_handle(messages, mcp_sessions):
    for item in messages[-1]["content"]:
        if item["type"] == "tool_use":
            signature = ", ".join("=".join((k, repr(v))) for k, v in item["input"].items())
            printer("tool_call", f"{item['name']}({signature})")
            approval = input("approve? (y/n): ").lower().strip()
            if approval != 'y':
                result_text = "ERROR: The user (the one you are chatting to) selected 'Deny' in the tool runner confirmation dialog"
                printer("tool_result", result_text)
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": item["id"],
                        "content": result_text,
                    }],
                })
                break
            session = mcp_sessions[item["name"].split("__")[0]]
            result = await session.call_tool(item["name"].split("__")[-1], item["input"])
            result_text = "".join([x.text for x in result.content])
            if result.isError:
                result_text = "ERROR: " + result_text
            printer("tool_result", result_text)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": item["id"],
                    "content": result_text,
                }],
            })

    return messages

async def llm(client, messages, tools, mcp_sessions):
    for _ in range(5):
        final_message = await claude(client, messages, tools)
        messages.append({"role": "assistant", "content": final_message["content"]})
        messages = await tools_handle(messages, mcp_sessions)
        if final_message["stop_reason"] != "tool_use":
            break

    return messages

async def mcp_session_start(prefix, mcp_config, exit_stack):
    stdio = stdio_client(StdioServerParameters(**mcp_config))
    read, write = await exit_stack.enter_async_context(stdio)
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    tools = await session.list_tools()

    return session, [{
        "name": prefix + x.name,
        "description": x.description,
        "input_schema": x.inputSchema
    } for x in tools.tools]

async def query_get():
    session = PromptSession()
    while True:
        with patch_stdout():
            prompt = await session.prompt_async('> ', prompt_continuation=lambda *a, **kw: "", multiline=True)

            return prompt.strip()

async def app(mcp_configs):
    async with AsyncExitStack() as exit_stack:
        client = await exit_stack.enter_async_context(AsyncAnthropic())

        tools, mcp_sessions = [], {}
        for prefix, mcp_config in mcp_configs.items():
            session, session_tools = await mcp_session_start(prefix + "__", mcp_config, exit_stack)
            mcp_sessions[prefix] = session
            tools.extend(session_tools)

        try:
            messages = []
            while True:
                query = await query_get()
                messages.append({"role": "user", "content": query})
                messages = await llm(client, messages, tools, mcp_sessions)
                print()
        except Exception as e:
            print("WHAT?", e)
            logger.exception("NOP")

def main():
    parser = argparse.ArgumentParser(
        description='A terminal LLM chat app with support for langchain tools and mcp servers')
    parser.add_argument('--mcp-config-file', type=str, help='Path to MCP config JSON file', required=True)
    args = parser.parse_args()

    with open(args.mcp_config_file) as f:
        mcp_configs = json.load(f)["mcpServers"]
        for mcp_config in mcp_configs.values():
            if "env" not in mcp_config:
                mcp_config["env"] = {}
            if "PATH" not in mcp_config["env"]:
                mcp_config["env"]["PATH"] = os.environ["PATH"]

    try:
        asyncio.run(app(mcp_configs), debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(e)
        logger.exception("ERROR")

if __name__ == "__main__":
    main()
