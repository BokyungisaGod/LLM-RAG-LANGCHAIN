from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    question: str
    질문은_인사가_맞나: bool


def 시작(state: GraphState) -> GraphState:
    print("시작이다!!")
    return state


def 질문이_인사인가(state: GraphState) -> GraphState:
    nowQuestion = state["question"]
    인사맞나 = False
    if nowQuestion == "안녕!!":
        print("질문이 인사다!! 맞음. state에 설정 해야지")
        인사맞나 = True

    state["질문은_인사가_맞나"] = 인사맞나
    return state


def 인사부터해야지(state: GraphState) -> GraphState:
    print("인사부터 해야지!!")
    return state


workflow = StateGraph(GraphState)
workflow.add_node("시작", 시작)
workflow.add_node("질문이_인사인가", 질문이_인사인가)
workflow.add_node("인사부터해야지", 인사부터해야지)


workflow.add_edge("시작", "질문이_인사인가")


def next_step(state: GraphState) -> bool:    
    return state["질문은_인사가_맞나"]


workflow.add_conditional_edges(
    "질문이_인사인가",
    next_step,
    {
        True: END,
        False: "인사부터해야지"
    },
)

workflow.set_entry_point("시작")
app = workflow.compile()

import pprint
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "TODO"})
inputs = GraphState(question="안녕!!", 질문은_인사가_맞나=False)

try:
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            pprint.pprint("===" * 10)
except GraphRecursionError as e:
    pprint.pprint(f"Recursion limit reached: {e}")
