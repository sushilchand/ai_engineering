Create a multi_agent and single_agent.

For single agent there will one ask_llm and 2 tools (keys available in env file)
1. calculator: it will be responsible for all the mathematical calculation
2. web_search: search the web using APIFY present in env file

The input for this will be a generic string which will contain the prompt about what needs to be done and output will be an answer along with number of LLM calls and token usage

Then Create a multi agent where there will be 3 llm calls
1. ask_manager_llm: it will not have any access to any tool but it will use to decide which llm to call based on the prompt. System prompt will be used in this so that manager's role is explained properly. This llm will also keep a note with it to keep a track of what task and response it gave to other llms
2. calculator: it will be responsible for all the mathematical calculation
3. web_search: search the web using APIFY present in env file

Make sure all the llm calls have maximum limit of 5 calls and it should fail with proper error once reached

Write the code for this without langgraph and it should be properly maintained withing classes
