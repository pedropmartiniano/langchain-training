from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from third_parties.twitter import scrape_user_tweets
from agents.linkedin_lookup_agent import linkedin_lookup_agent
from agents.twitter_loopkup_agent import twitter_lookup_agent
from dotenv import load_dotenv
from output_parsers import summary_parser, Summary
from typing import Tuple

def ice_break_with(name: str) -> Tuple[Summary, str]:
    linkedin_url = linkedin_lookup_agent(name + " linkedin")
    twitter_user = twitter_lookup_agent(name + " twitter")

    linkedin_data = scrape_linkedin_profile(linkedin_url)
    tweets = scrape_user_tweets(twitter_user)
    
    summary_template = """
        given linkedin the information about a person from linkedin {information},
        and their latest twitter posts {twitter_posts}, I want you to create:
        1. a short summary
        2. two interesting facts about them
        
        Use both information from linkedin and twitter
        \n{format_instructions}
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information", "twitter_posts"],
        template=summary_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5
    )

    chain = summary_prompt_template | llm | summary_parser
    
    res: Summary = chain.invoke(input={"information": linkedin_data, "twitter_posts": tweets})
    
    return res, linkedin_data.get('profile_pic_url')

if __name__ == "__main__":
    load_dotenv()
    print("Ice Breaker Enter")
    print(ice_break_with("Pedro Paulino Martiniano"))
