from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub
from tools.tools import get_profily_url_tavily

load_dotenv()


def twitter_lookup_agent(name: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
        ) # creating the LLM models that the agent will use to generate the answer

    template = """
        given the full name and some additional information {name_of_person} I want you to find a link to their twitter profile page, and extract from it their username.
        IMPORTANT: Your answer should contain only the username of the Twitter profile page and nothing more.
        Response format: <twitter-username>
    """ # creating the template for the prompt using a dinamic variable that will be used in the prompt

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    ) # creating the prompt template for the agent passing the input variables that will be used dinamically in the prompt
    
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 twitter profile page",
            func=get_profily_url_tavily,
            description="Useful for when you need get the Twitter Page URL" # description is the field used to the agent determine if the tool is useful for the task
        ) # creating the search tool
    ] # all the tools that the agent might use to get the answer
    
    react_prompt = hub.pull("hwchase17/react") # its going to be the reasoning engine of the agent
    
    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt
    )
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True
    )
    
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    
    linkedin_profile_url = result["output"]
    return linkedin_profile_url

if __name__ == "__main__":
    name = "Pedro Paulino Martiniano"
    linkedin_profile_url = twitter_lookup_agent(name)

    print(linkedin_profile_url)