"""
Price scraper for EA FC 26 Players from Futbin website. Used for week 1 and week 2 scrapes.
"""

import time
import re
import os

import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {"User-Agent": "Mozilla/5.0"}

# Take URLs from the scraped list
df_urls = pd.read_csv("/files/Capstone_Project_ST/data/player_urls.csv")

df_urls = df_urls.head(5) # Test for little amount of players

rows = []
total_players = len(df_urls)

for index, url in enumerate(df_urls["url"], start=1):
    print(f"Scraping price for player {index}/{total_players}: {url}")
    
    try:
        resp = requests.get(url, headers=headers, timeout=10) # So the program does not run indefinitely in case of connection problems
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        row = {"url": url}
        
        # Finds lowest price of each player
        price_el = soup.select_one("div.price.inline-with-icon.lowest-price-1")
        if price_el:
            text_price = price_el.get_text(strip=True)
            digits = re.sub(r"[^\d]", "", text_price) # So only digits appear
            row["price"] = int(digits) if digits else None
        else:
            row["price"] = None
        
        rows.append(row)
        time.sleep(1) # Avoid too many requests in a short time
        
        # Ackowledge if there was an error with a specific player
    except requests.exceptions.RequestException as e:
        print(f"Error on {url}: {e}")
        row = {"url": url, "price": None}
        rows.append(row)
        continue

# Save player prices to week1 folder (CSV file)
output_dir = "/files/Capstone_Project_ST/data/week1"
os.makedirs(output_dir, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(f"{output_dir}/prices_week1.csv", index=False)
print(f"Scraped {len(df)} player prices to {output_dir}/prices_week1.csv")
