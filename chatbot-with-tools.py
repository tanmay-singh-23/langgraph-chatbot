from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()

class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearch(max_results=2, topic="general")

tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant")

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: BasicChatBot):
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tool_node"
    
    return END

tool_node = ToolNode(tools = tools)

graph = StateGraph(BasicChatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_router, {"tool_node": "tools", END:END})
graph.add_edge("tools", "chatbot")

app = graph.compile()

while True:
    user_input = input("User: ")

    if user_input in ["exit", "end"]:
        break
    else:
        response = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        print(response)

# print(result)

