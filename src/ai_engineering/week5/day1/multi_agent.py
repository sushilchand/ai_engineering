import json
from typing import Any

from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME
from ai_engineering.week5.day1.calculator import CALCULATOR_TOOL, calculator
from ai_engineering.week5.day1.web_search import WEB_SEARCH_TOOL, web_search


class AgentCallLimitError(RuntimeError):
    """Raised when an agent exceeds the maximum number of LLM calls."""


class BaseAgent:
    def __init__(self, model_name: str | None = None, max_calls: int = 10):
        self.model_name = model_name or LLM_MODEL_NAME
        self.client = Groq(api_key=GROQ_API_KEY)
        self.max_calls = max_calls
        self.llm_calls = 0
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _check_call_limit(self) -> None:
        if self.llm_calls >= self.max_calls:
            raise AgentCallLimitError(
                f"Maximum LLM call limit reached ({self.max_calls})."
            )

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

        usage = getattr(response, "usage", None)
        if usage:
            self.token_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.token_usage["completion_tokens"] += (
                getattr(usage, "completion_tokens", 0) or 0
            )
            self.token_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

        return response.choices[0].message


class CalculatorAgent(BaseAgent):
    def __init__(self, model_name: str | None = None, max_calls: int = 5):
        super().__init__(model_name=model_name, max_calls=max_calls)

    def run(self, prompt: str) -> dict[str, Any]:
        system_prompt = (
            "You are a calculator specialist. Evaluate arithmetic problems exactly and "
            "use the calculator tool whenever needed. Return the final result in plain text. Always answer in 2 linex maximum"
        )
        messages = [{"role": "user", "content": prompt}]
        assistant_message = self.ask_llm(
            messages, system_prompt=system_prompt, tools=[CALCULATOR_TOOL]
        )

        if not getattr(assistant_message, "tool_calls", None):
            return {
                "answer": assistant_message.content or "",
                "llm_calls": self.llm_calls,
                "token_usage": self.token_usage,
            }

        tool_results = []
        for tool_call in assistant_message.tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = calculator(**arguments)
            tool_results.append(str(result))

        return {
            "answer": "\n".join(tool_results),
            "llm_calls": self.llm_calls,
            "token_usage": self.token_usage,
        }


class WebSearchAgent(BaseAgent):
    def __init__(self, model_name: str | None = None, max_calls: int = 5):
        super().__init__(model_name=model_name, max_calls=max_calls)

    def run(self, prompt: str) -> dict[str, Any]:
        system_prompt = (
            "You are a research assistant. Use the web_search tool to gather current "
            "facts and provide a concise answer based on the search results. Always answer in 2 linex maximum"
        )
        messages = [{"role": "user", "content": prompt}]
        assistant_message = self.ask_llm(
            messages, system_prompt=system_prompt, tools=[WEB_SEARCH_TOOL]
        )

        if not getattr(assistant_message, "tool_calls", None):
            return {
                "answer": assistant_message.content or "",
                "llm_calls": self.llm_calls,
                "token_usage": self.token_usage,
            }

        tool_results = []
        for tool_call in assistant_message.tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            result = web_search(**arguments)
            tool_results.append(str(result))

        return {
            "answer": "\n\n".join(tool_results),
            "llm_calls": self.llm_calls,
            "token_usage": self.token_usage,
        }


class MultiAgent(BaseAgent):
    def __init__(self, model_name: str | None = None, max_calls: int = 5):
        super().__init__(model_name=model_name, max_calls=max_calls)
        self.manager_system_prompt = (
            "You are the manager of a multi-agent system. You are never allowed to call tools. "
            "Your job is to decide which specialist llm to call next based on the user prompt and the "
            "latest specialist response. Keep reading the response and deciding until the task is fully "
            "answered. Return valid JSON only with keys: 'next_agent' and 'reason'. Allowed values for "
            "'next_agent' are 'calculator', 'web_search', or 'final'. Use 'final' only when the answer "
            "is ready to return. Always answer in 2 linex maximum"
        )
        self.task_log: list[str] = []
        self.calculator_agent = CalculatorAgent(
            model_name=model_name, max_calls=max_calls
        )
        self.web_search_agent = WebSearchAgent(
            model_name=model_name, max_calls=max_calls
        )

    def ask_manager_llm(self, prompt: str, specialist_results: list[str]) -> str:
        context = prompt
        if specialist_results:
            context = f"{prompt}\n\nSpecialist responses so far:\n" + "\n---\n".join(
                specialist_results
            )

        message = self.ask_llm(
            [{"role": "user", "content": context}],
            system_prompt=self.manager_system_prompt,
        )
        content = (message.content or "").strip()

        try:
            parsed = json.loads(content)
            route = str(parsed.get("next_agent", "final")).strip().lower()
            reason = str(parsed.get("reason", ""))
            self.task_log.append(
                f"Manager decision: next_agent={route} | reason={reason}"
            )
            return self._normalize_route(route)
        except (TypeError, ValueError):
            route = self._normalize_route(content)
            self.task_log.append(f"Manager decision: next_agent={route}")
            return route

    def _normalize_route(self, route: str) -> str:
        route = route.strip().lower()
        if route.startswith("{"):
            try:
                parsed = json.loads(route)
                route = str(
                    parsed.get("next_agent", parsed.get("route", "final"))
                ).lower()
            except json.JSONDecodeError:
                pass
        if route not in {"calculator", "web_search", "final"}:
            if "calculator" in route:
                return "calculator"
            if "search" in route or "web" in route:
                return "web_search"
            return "final"
        return route

    def _finalize_answer(self, prompt: str, specialist_results: list[str]) -> str:
        if self.llm_calls >= self.max_calls:
            raise AgentCallLimitError(
                f"Maximum LLM call limit reached ({self.max_calls}) before final synthesis."
            )

        context = f"Original prompt:\n{prompt}\n\nSpecialist results:\n"
        if specialist_results:
            context += "\n---\n".join(specialist_results)

        response = self.ask_llm(
            [{"role": "user", "content": context}],
            system_prompt=(
                "You are the final answer synthesizer. Use the specialist results to answer the user's "
                "prompt accurately. If the prompt contains multiple questions, answer all parts. "
                "Do not call any tools. Keep the answer concise but complete."
            ),
        )
        return response.content or "\n---\n".join(specialist_results)

    def run(self, prompt: str) -> dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        specialist_results: list[str] = []
        total_specialist_llm_calls = 0
        total_specialist_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _ in range(self.max_calls):
            route = self.ask_manager_llm(prompt, specialist_results)

            if route == "final":
                final_answer = self._finalize_answer(prompt, specialist_results)
                return {
                    "route": "final",
                    "answer": final_answer,
                    "llm_calls": self.llm_calls + total_specialist_llm_calls,
                    "token_usage": {
                        "prompt_tokens": self.token_usage["prompt_tokens"]
                        + total_specialist_tokens["prompt_tokens"],
                        "completion_tokens": self.token_usage["completion_tokens"]
                        + total_specialist_tokens["completion_tokens"],
                        "total_tokens": self.token_usage["total_tokens"]
                        + total_specialist_tokens["total_tokens"],
                    },
                    "task_log": self.task_log,
                }

            if route == "calculator":
                result = self.calculator_agent.run(prompt)
                specialist_results.append(result["answer"])
                total_specialist_llm_calls += result["llm_calls"]
                total_specialist_tokens["prompt_tokens"] += result["token_usage"][
                    "prompt_tokens"
                ]
                total_specialist_tokens["completion_tokens"] += result["token_usage"][
                    "completion_tokens"
                ]
                total_specialist_tokens["total_tokens"] += result["token_usage"][
                    "total_tokens"
                ]
                continue

            if route == "web_search":
                result = self.web_search_agent.run(prompt)
                specialist_results.append(result["answer"])
                total_specialist_llm_calls += result["llm_calls"]
                total_specialist_tokens["prompt_tokens"] += result["token_usage"][
                    "prompt_tokens"
                ]
                total_specialist_tokens["completion_tokens"] += result["token_usage"][
                    "completion_tokens"
                ]
                total_specialist_tokens["total_tokens"] += result["token_usage"][
                    "total_tokens"
                ]
                continue

            raise AgentCallLimitError(
                f"MultiAgent reached the maximum LLM call limit of {self.max_calls}."
            )

        raise AgentCallLimitError(
            f"MultiAgent reached the maximum LLM call limit of {self.max_calls}."
        )


if __name__ == "__main__":
    agent = MultiAgent()
    sample_prompt = "What is the height of burj khalifa and how many eiffel towers can be stacked in burj khalifa based on its height ?"
    result = agent.run(sample_prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
