from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import END, StateGraph

class GraphState(TypedDict):
    question: str
    질문은_인사가_맞나: bool
    악수했음: bool
    status: str   # stand, walk, run, seat, die


def 시작(state: GraphState) -> GraphState:
    return state


def 인사부터해야지(state: GraphState) -> GraphState:
    print("인사부터 해야지!!")
    return state



def 질문이_인사인가(state: GraphState) -> GraphState:    
    nowQuestion = state["question"]    
    인사맞나 = False    
    if nowQuestion == "인사인가?":    
        인사맞나 = True
    else:
        인사맞나 = False
    
    state["질문은_인사가_맞나"] = 인사맞나        
    return state

def 나도망간다(state: GraphState) -> GraphState:
    print("나도 망간다")
    return state

def 잘했어(state: GraphState) -> GraphState:
    print("잘했어")
    return state

## 성민님 영역


def 인사아님(state: GraphState) -> GraphState:
        
    print("인사부터 해라 ㅅㄱ")
    print("인사? 그거 중요하지. 사람 만나면 인사부터 해야지. 고대에는 인사가 적대감 없다는 신호였어. 요즘도 문화마다 다르지만, 기본은 똑같아. 서양은 악수, 동양은 고개 숙임. 어쨌든 인사는 필수지. 안 하면 무례한 사람 되는 거야.")
    return state



## 유현님 영역
def Handshake(state: GraphState) -> GraphState:
    DidHandshake = state["악수했음"] # ???
    Hands = False

    if DidHandshake == "악수함 ㅇㅇ":
        Hands = True

    state["악수했음"] = Hands
    return state
    


##
def 도망감(state: GraphState) -> GraphState:
    인사는했나요 = state["질문은_인사가_맞나"]
    악수는했나요 = state["악수했음"]
    return state



workflow = StateGraph(GraphState)
workflow.add_node("시작", 시작)
workflow.add_node("is_insa", 질문이_인사인가)
workflow.add_node("insa_anya", 인사아님)
workflow.add_node("Handshake", Handshake)
workflow.add_node("도망감", 도망감)

#edge 코딩하기!!
workflow.add_edge("시작","insa_anya")
workflow.add_edge("insa_anya","Handshake")
workflow.add_edge("Handshake","도망감")

def next_step(state: GraphState) -> GraphState:    
    return state["질문은_인사가_맞나"]

#질문은_인사가_맞나: bool
#악수했음: bool

workflow.add_conditional_edges(
    "도망감",
    next_step,
    {
        True:"나도망간다",
        False:"잘했어"
    },
)





workflow.set_entry_point("시작")
app = workflow.compile()


workflow.add_conditional_edges(
    "is_insa",
    next_step,
    {
        True:"Handshake",
        False:"인사아님"
    }
)