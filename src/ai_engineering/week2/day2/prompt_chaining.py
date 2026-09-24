from time import sleep

from groq import Groq

from ai_engineering.constants import GROQ_API_KEY, LLM_MODEL_NAME


class PromptChaining:
    def __init__(self, jd: str, resume: str):
        self.jd = jd
        self.resume = resume
        self.client = Groq(api_key=GROQ_API_KEY)

    def run_llp(self, sys_prompt: str, user_prompt: str) -> str:
        sys_msg = {"role": "system", "content": sys_prompt}

        user_msg = {"role": "user", "content": user_prompt}

        messages = [sys_msg, user_msg]
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME, messages=messages
        )
        sleep(2)
        return response.choices[0].message.content

    def parse_jd(self) -> str:
        sys_prompt = """
            You are an expert HR assistant and extract the skills from the JD
        """
        user_prompt = f"""
            Extract the skills from {self.jd} and make sure you extract the skills in a list of comma separated values and don't add anything else in response
        """
        result = self.run_llp(sys_prompt=sys_prompt, user_prompt=user_prompt)
        print(f"Step 1 -------\n Result: {result}")
        return result

    def parse_resume(self) -> str:
        sys_prompt = """
            You are an expert HR assistant and extract the skills from the Resume
        """
        user_prompt = f"""
            Extract the skills from {self.resume} and make sure you extract the skills in a list of comma separated values and don't add anything else in response
        """
        result = self.run_llp(sys_prompt=sys_prompt, user_prompt=user_prompt)
        print(f"Step 2 -------\n Result: {result}")
        return result

    def calculate_score(self, resume_skills: str, jd_skills: str) -> str:
        sys_prompt = """
            You are an expert HR assistant and calculate a score strictly between 1 to 100 based on matching. Make sure you are returing only score and verdict and nothing else
        """
        user_prompt = f"""
            Compare {resume_skills} and {jd_skills} and calculate a score and a verdict based on how much skills are actually present in {resume_skills} which are required in {jd_skills}.
        """
        result = self.run_llp(sys_prompt=sys_prompt, user_prompt=user_prompt)
        print("Step 3 -------")
        return result


def main():
    JD = """
    We are hiring a Backend Python Developer.

    Requirements:
    - Strong Python
    - FastAPI or Django
    - PostgreSQL
    - Docker
    - AWS
    - REST APIs
    - 2+ years of experience
    """
    RESUME = """
    Name: Rahul Sharma

    Experience:
    3 years as a Software Developer.

    Skills:
    Python, FastAPI, PostgreSQL, Docker, Django
    REST APIs, Git, Docker, amazon web service

    Projects:
    Built a food delivery backend using
    FastAPI and MySQL.

    Deployed applications using Docker.
    """

    prompt_chaining = PromptChaining(jd=JD, resume=RESUME)
    required_skills = prompt_chaining.parse_jd()
    available_skills = prompt_chaining.parse_resume()
    final_answer = prompt_chaining.calculate_score(
        resume_skills=available_skills, jd_skills=required_skills
    )
    print(final_answer)


if __name__ == "__main__":
    main()
