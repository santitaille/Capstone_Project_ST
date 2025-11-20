"""
URL scraper for EA FC 26 Players from Futbin website.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Manually setted up Futbin website to find the URL for list of all 83+ OVR non-goalkeepers tradeable rated players
BASE_URL = (
    "https://www.futbin.com/players?page={}&ps_price=200%2B&"
    "position=LB%2CCB%2CRB%2CCAM%2CCM%2CCDM%2CRM%2CLM%2CST%2CRW%2CLW&"
    "pos_type=all&player_rating=83-99&eUnt=1"
)
TOTAL_PAGES = 2 # Checked first for small amount of pages
OUTPUT_FILE = "/files/Capstone_Project_ST/data/player_urls.csv"

def scrape_player_urls(base_url: str, total_pages: int):
    """Scrape player URLs from Futbin website."""
    all_urls = []
    failed_pages = []

    for page in range(1, total_pages + 1):
        url = base_url.format(page)
        print(f"Scraping page {page}/{total_pages}")
        
        try:
            response = requests.get(url, timeout=10) # So the program does not run indefinitely in case of connection problems
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find player URLs from each table row
            rows = soup.select("table.futbin-table.players-table tbody tr")
            for row in rows:
                a_tag = row.select_one("td.table-name a")
                if a_tag and a_tag.get("href"):
                    full_url = "https://www.futbin.com" + a_tag.get("href")
                    all_urls.append(full_url)

        # Ackowledge if there was an error
        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
            failed_pages.append(page)
            continue
    
    if failed_pages:
        print(f"{failed_pages} pages failed to scrape.")
    
    return all_urls

# Scrape and save URLs to CSV file
if __name__ == "__main__":
    urls = scrape_player_urls(BASE_URL, TOTAL_PAGES)
    df = pd.DataFrame({"url": urls})
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Scraped {len(urls)} URLs")