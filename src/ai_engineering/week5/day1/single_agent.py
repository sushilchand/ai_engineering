import json
from dataclasses import dataclass
from typing import Any

from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME
from ai_engineering.week5.day1.calculator import CALCULATOR_TOOL, calculator
from ai_engineering.week5.day1.web_search import WEB_SEARCH_TOOL, web_search


class AgentCallLimitError(RuntimeError):
    """Raised when an agent exceeds the maximum number of LLM calls."""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return

        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class BaseAgent:
    def __init__(self, model_name: str | None = None, max_calls: int = 5):
        self.model_name = model_name or LLM_MODEL_NAME
        self.client = Groq(api_key=GROQ_API_KEY)
        self.max_calls = max_calls
        self.llm_calls = 0
        self.token_usage = TokenUsage()

    def _check_call_limit(self) -> None:
        if self.llm_calls >= self.max_calls:
            raise AgentCallLimitError(
                f"Maximum LLM call limit reached ({self.max_calls})."
            )

    def _track_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.token_usage.add(usage)

    def ask_llm(
        self,
        messages: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        self._check_call_limit()
        payload = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            tools=tools,
            tool_choice="auto" if tools else "none",
            temperature=0.2,
        )
        self.llm_calls += 1
        self._track_usage(response)
        return response.choices[0].message


class SingleAgent(BaseAgent):
    def __init__(self, model_name: str | None = None, max_calls: int = 5):
        super().__init__(model_name=model_name, max_calls=max_calls)
        self.tools = [CALCULATOR_TOOL, WEB_SEARCH_TOOL]
        self.tool_functions = {
            "calculator": calculator,
            "web_search": web_search,
        }

    def run(self, prompt: str) -> dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.max_calls):
            assistant_message = self.ask_llm(messages, tools=self.tools)
            messages.append(assistant_message.model_dump(exclude_none=True))

            if not getattr(assistant_message, "tool_calls", None):
                answer = assistant_message.content or ""
                return {
                    "answer": answer,
                    "llm_calls": self.llm_calls,
                    "token_usage": self.token_usage.as_dict(),
                }

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments or "{}")
                try:
                    result = self.tool_functions[tool_name](**arguments)
                except Exception as error:
                    result = f"Tool error: {error}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        raise AgentCallLimitError(
            f"SingleAgent reached the maximum LLM call limit of {self.max_calls}."
        )


if __name__ == "__main__":
    agent = SingleAgent()
    sample_prompt = "What is the height of burj khalifa and how many eiffel towers can be stacked in burj khalifa based on its height ?"
    result = agent.run(sample_prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
