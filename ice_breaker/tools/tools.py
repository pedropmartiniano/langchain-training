from langchain_community.tools.tavily_search import TavilySearchResults

def get_profily_url_tavily(search_str: str):
    """
        Searchs for Linkedin profile URL using Tavily Search
    """
    
    search = TavilySearchResults()
    res = search.run(f"{search_str}")

    return res
