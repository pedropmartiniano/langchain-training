import requests
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """
        scrape information from linkedin profiles,
        Manually scrape the information from the linkedin profile.
    """
    
    if mock:
        linkedin_profile_url = "https://gist.github.com/pedropmartiniano/4439b5dd55281b3b1f57f1f4775330c1/raw/pedro-paulino.json"
        response = requests.get(linkedin_profile_url, timeout=10)
    else:
        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
        headers = {
            "Authorization": f"Bearer {os.getenv('PROXYCURL_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            api_endpoint,
            headers=headers,
            params={"url": linkedin_profile_url},
            timeout=10
        )
    
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], None, "")
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get('groups'):
        for group_dict in data.get('groups'):
            group_dict.pop("profile_pic_url", None)

    return data

if __name__ == "__main__":
    print(scrape_linkedin_profile("https://www.linkedin.com/in/pedro-paulino-martiniano/", True))
