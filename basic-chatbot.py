from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from IPython.display import Image, display
from dotenv import load_dotenv

from pprint import pprint

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant")

class BasicChatState(TypedDict):
    messages : Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
# graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

print(app.get_graph().draw_mermaid())
# display(Image(app.get_graph().draw_mermaid_png()))

while True:
    user_input = input("User: ")

    if user_input in ["exit", "end"]:
        break
    else:
        response = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        # print(response)
        pprint(response)
