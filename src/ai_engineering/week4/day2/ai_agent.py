import ast
import json
import math
import operator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from groq import Groq

from ai_engineering.constants import APIFY_KEY, GROQ_API_KEY, LLM_MODEL_NAME


def web_search(query: str) -> str:
    if not APIFY_KEY:
        raise RuntimeError("APIFY_KEY is not configured")

    endpoint = "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items"
    url = f"{endpoint}?{urlencode({'token': APIFY_KEY})}"
    payload = json.dumps({"queries": query, "maxPagesPerQuery": 1}).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            search_data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Apify web search failed: {error}") from error

    results = [
        result for page in search_data for result in page.get("organicResults", [])
    ]
    if not results:
        return "No search results found."

    return "\n\n".join(
        f"{result.get('title', 'Untitled')}\n"
        f"{result.get('url', '')}\n"
        f"{result.get('description', '')}"
        for result in results[:5]
    )


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_arithmetic(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_arithmetic(node.left)
        right = _evaluate_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent must be between -1000 and 1000")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_arithmetic(node.operand))
    raise ValueError("Only basic arithmetic expressions are supported")


def calculate(expression: str) -> str:
    if len(expression) > 256:
        raise ValueError("Expression is too long")

    try:
        result = _evaluate_arithmetic(ast.parse(expression, mode="eval").body)
    except (ArithmeticError, SyntaxError, TypeError) as error:
        raise ValueError(f"Invalid arithmetic expression: {expression}") from error

    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("The result must be a finite number")
    return str(result)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet and return relevant results.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_FUNCTIONS = {"web_search": web_search, "calculate": calculate}


def run_agent(prompt: str, max_steps: int = 5) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Give answer only in maximum 2 lines"},
        {"role": "user", "content": prompt},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
                result = _TOOL_FUNCTIONS[tool_name](**arguments)
            except Exception as error:
                result = f"Tool error: {error}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return "The agent reached the maximum number of tool calls."


if __name__ == "__main__":
    print(
        run_agent(
            "List down top 10 countries with maximum population density and also list down the numbers"
        )
    )
