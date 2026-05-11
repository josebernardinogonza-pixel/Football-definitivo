import os
import requests

API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

def fetch_fixtures():
    url = "https://v3.football.api-sports.io/fixtures?league=39&season=2023"
    headers = {'x-apisports-key': API_FOOTBALL_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['response']

def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds?apiKey={ODDS_API_KEY}&regions=uk&markets=h2h"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
