import re
import time

from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME
from ai_engineering.week2.day6.utils import calculator, get_pricing


def main():
    client = Groq(api_key=GROQ_API_KEY)
    model = LLM_MODEL_NAME

    tools = {"get_pricing": get_pricing, "calculator": calculator}

    sys_prompt = """
    You are a shopping assistant

    You have the following tools
    get_pricing(options)
    calculater(expressioin)

    Call tool exactly like this example
    Action: get_pricing("iphone 17 pro")
    Action: calculator(5000-1000)

    Never write:
    get_pricing("5000-1000")

    Never write:
    calculater("iphone 17 pro")

    Strictly follow these rules:
    1. Decide what needs to be done next
    2. Call only one tool at a time
    3. Afer an action stop immediately
    4. Wait until you receive observation
    5. Never invent new tool or guess observation
    6. Then decide what to do next
    7. When the task is complete, give Final answer

    Format:
    Thought: what needs to be done
    Action: tool_name(arguement)

    When finished:
    Final Answer: your answer
    """

    def run_agent(prompt, total_steps=5) -> str:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        for step in range(total_steps):
            print(f"---------------\nStep {step}\n--------------\n")
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0
            )
            answer = response.choices[0].message.content

            if "Final Answer" in answer:
                return answer.split(":")[1].strip()

            print(answer)

            match = re.search(r"Action:\s*(\w+)\((.*?)\)", answer)

            if match:
                tool_name = match.group(1)
                tool_input = match.group(2).strip().strip('"')

                if tool_name in tools:
                    observation = tools[tool_name](tool_input)
                else:
                    observation = "Tool not found"

                print(f"Observation: {observation}")

                messages.append({"role": "assistant", "content": answer})

                messages.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
                time.sleep(5)

    user_prompt = """
    I want to buy either iphone 17 or iphone 17 pro, based on the pricing tell my how many I can buy. I have 1000000 budget.
    """
    result = run_agent(prompt=user_prompt, total_steps=20)
    print(f"Final Result is : {result}")


if __name__ == "__main__":
    main()
