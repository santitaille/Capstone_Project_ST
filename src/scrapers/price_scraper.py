"""
Price Scraper for EA FC 26 Players from Futbin.

Scrapes current market prices for players. Run separately for Week 1 and Week 2.

Generates:
* Table: Player prices for specified week (CSV)

IMPORTANT:
For second scrape: uncomment lines marked with #W2 and comment lines marked with #W1.
"""

import time
import re
import os
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

headers = {"User-Agent": "Mozilla/5.0"}

# Configuration
INPUT_FILE = "/files/Capstone_Project_ST/data/player_urls.csv"
OUTPUT_DIR = "/files/Capstone_Project_ST/data/week2"  # Change week1 to week2
OUTPUT_FILE = "prices_week2.csv"  # Change week1 to week2

# Had to exclude some players (removed from market after wrong inclusion)
# So those players won't be part of the price scraping
EXCLUDE_URLS = [
    "https://www.futbin.com/26/player/52/alexander-isak",
    "https://www.futbin.com/26/player/398/piero-hincapie",
    "https://www.futbin.com/26/player/17322/benjamin-pavard",
    "https://www.futbin.com/26/player/20451/victor-osimhen",
    "https://www.futbin.com/26/player/20452/micky-van-de-ven",
    "https://www.futbin.com/26/player/20577/troy-parrott",
    "https://www.futbin.com/26/player/20453/mikel-gogorza",
    "https://www.futbin.com/26/player/20454/dan-burn",
]

# Take URLs from the scraped list, takes out excluded players
df_urls = pd.read_csv(INPUT_FILE)
df_urls = df_urls[~df_urls["url"].isin(EXCLUDE_URLS)]

rows = []
failed_players = []

total_players = len(df_urls)

if __name__ == "__main__":
    logger.info("Starting price scraper")

    for index, url in enumerate(df_urls["url"], start=1):
        logger.info("Scraping price for player %d/%d", index, total_players)

        # So the program does not run indefinitely in case of connection problems
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            row = {"url": url}

            # Finds lowest price of each player
            price_el = soup.select_one("div.price.inline-with-icon.lowest-price-1")
            if price_el:
                text_price = price_el.get_text(strip=True)
                digits = re.sub(r"[^\d]", "", text_price)  # So only digits appear
                row["price"] = int(digits) if digits else None
            else:
                row["price"] = None

            rows.append(row)
            time.sleep(1)  # Avoid too many requests in a short time

            # Acknowledge if there was an error with a specific player
        except requests.exceptions.RequestException as e:
            logger.error("Error on player %d: %s", index, e)
            row = {"url": url, "price": None}
            rows.append(row)
            failed_players.append(index)
            continue

    # Acknowledge the number of players that failed to scrape
    if failed_players:
        logger.warning("%d players failed to scrape", len(failed_players))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    OUTPUT_PATH = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Scraped %d player prices to %s", len(df), OUTPUT_PATH)
