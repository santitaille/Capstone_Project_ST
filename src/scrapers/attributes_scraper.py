"""
Attributes scraper for EA FC 26 Players from Futbin website. Attributes remain constant throught the game.
Attributes collected include name, rating, pace, shooting, passing, dribbling, defending, physical, weak foot, skill moves and other relevant attributes.
Following similar strcuture to url_scraper.py and price_scraper.py
"""

import time

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Configuration
INPUT_FILE = "/files/Capstone_Project_ST/data/player_urls.csv"
OUTPUT_DIR = "/files/Capstone_Project_ST/data"
OUTPUT_FILE = "players_attributes.csv"

headers = {"User-Agent": "Mozilla/5.0"}

def get_text(el):
    return el.get_text(strip=True) if el else None

# Read player URLs
df_urls = pd.read_csv(INPUT_FILE)

df_urls = df_urls.head(7)  # Testing first for small amount of players

rows = []
total_players = len(df_urls)

for index, url in enumerate(df_urls["url"], start=1):
    print(f"Scraping attributes for player {index}/{total_players}")
    
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    row = {"url": url}
    
    # extratc player name using its specific URL and its specific HTML structure
    name_el = soup.select_one("h1.page-header-top")
    if name_el:
        full_text = get_text(name_el)
        if " - " in full_text: #because the HTML strucuture is for example: player name - card rarity) so to only keep the player name
            player_name = full_text.split(" - ")[0].strip()
        elif " - " in full_text:
            player_name = full_text.split(" - ")[0].strip()
        else:
            player_name = full_text
        row["player_name"] = player_name.title()
    else:
        row["player_name"] = None
    
    rows.append(row)
    time.sleep(1) # Avoid too many requests in a short time

# Save to CSV file
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(rows)
output_path = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
df.to_csv(output_path, index=False)

print(f"Scraped {len(df)} player attributes to {output_path}")
