from typing import TypedDict

from langgraph.graph import END, StateGraph


class State(TypedDict):
    number: int


def double(state: State) -> dict:
    print(f"In double: {state['number']}")
    new_num = state["number"] * 2
    return {"number": new_num}


def finish(state: State) -> dict:
    print(f"In finish: {state['number']}")
    return {"number": state["number"]}


def decision(state: State) -> str:
    if state["number"] > 100:
        return "finish"
    else:
        return "double"


def main():
    graph = StateGraph(State)

    graph.add_node("double", double)
    graph.add_node("finish", finish)

    graph.set_entry_point("double")

    graph.add_conditional_edges(
        "double", decision, {"double": "double", "finish": "finish"}
    )

    graph.add_edge("finish", END)

    compiled_graph = graph.compile()

    compiled_graph.invoke({"number": 9, "log": ""})


if __name__ == "__main__":
    main()
