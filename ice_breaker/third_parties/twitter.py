from dotenv import load_dotenv
import requests

load_dotenv()

def scrape_user_tweets(username: str):
    """
        Scrapes a Twitter user's original tweets and returns them as a list of dictionaries.
        Each dictionary has three fields: "time_posted", "text" and "url";
    """
    
    url = "https://gist.github.com/pedropmartiniano/18cd06ca177becdf77825f2d732c82e2/raw/pedro-paulino-tweets.json"
    tweets = requests.get(url, timeout=5).json()
    
    tweets_list = []

    for tweet in tweets:
        tweet_dict = {}
        tweet_dict["text"] = tweet["text"]
        tweet_dict["time_posted"] = tweet["time_posted"]
        tweet_dict["url"] = f"https://twitter.com/pedropmartiniano/status/{tweet["id"]}"
        
        tweets_list.append(tweet_dict)
        
    return tweets_list

if __name__ == "__main__":
    print(scrape_user_tweets("pedropmartiniano"))