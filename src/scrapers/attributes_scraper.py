"""
Attributes scraper for EA FC 26 Players from Futbin website. Attributes remain constant throught the game.
Attributes collected include name, rating, pace, shooting, passing, dribbling, defending, physical, weak foot, skill moves and other relevant attributes.
Following similar strcuture to url_scraper.py and price_scraper.py
"""

import time
import os
import logging

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "/files/Capstone_Project_ST/data/player_urls.csv"
OUTPUT_DIR = "/files/Capstone_Project_ST/data"
OUTPUT_FILE = "players_attributes.csv"

headers = {"User-Agent": "Mozilla/5.0"}

# as for price scraper, had to exclude some players that were mistakenly inluded in the market for a very short time before being removed
# So those players won't be part of the price scraping
EXCLUDE_URLS = [
    "https://www.futbin.com/26/player/52/alexander-isak",
    "https://www.futbin.com/26/player/398/piero-hincapie",
    "https://www.futbin.com/26/player/17322/benjamin-pavard",
    "https://www.futbin.com/26/player/20451/victor-osimhen",
    "https://www.futbin.com/26/player/20452/micky-van-de-ven",
    "https://www.futbin.com/26/player/20577/troy-parrott",
    "https://www.futbin.com/26/player/20453/mikel-gogorza",
    "https://www.futbin.com/26/player/20454/dan-burn"
]

def get_text(el):
    """Extract text from HTML element with no errors (managing None values)."""
    return el.get_text(strip=True) if el else None

# Take URLs from the scraped list, takes out excluded players
df_urls = pd.read_csv(INPUT_FILE)
df_urls = df_urls[~df_urls['url'].isin(EXCLUDE_URLS)]

rows = []
failed_players = []
total_players = len(df_urls)

if __name__ == "__main__":
    logger.info("Starting attributes scraper")
    
    for index, url in enumerate(df_urls["url"], start=1):
        logger.info(f"Scraping attributes for player {index}/{total_players}")

        try:
            resp = requests.get(url, headers=headers, timeout=10)  # Gets HTML text of each player's page, and timout so the program does not run indefinitely in case of connection problems
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser") # convert HTML text into BeautifulSoup object
            card = soup.select_one("div.playercard-option-wrapper div.playercard-26") # finds player info card section
    
            row = {"url": url}
        
            # EXTRACT NAME
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
            
            # EXTRACT GENDER
            # gender is not directly specified on Futbin, however the descrption text contains pronouns
            player_text_section = soup.select_one("div.player-text-section")
            if player_text_section:
                text_content = player_text_section.get_text().lower()
                if " he " in text_content or text_content.startswith("he "):
                    row["gender"] = "M"
                elif " she " in text_content or text_content.startswith("she "):
                    row["gender"] = "F"
                else:
                    row["gender"] = None
            else:
                row["gender"] = None

            # EXTRACT OVR RATING
            rating_el = card.select_one("div.playercard-26-rating-pos-wrapper")
            if rating_el:
                rating_text = rating_el.get_text(strip=True)
                row["rating"] = int(rating_text[:2])

            # EXTRACT CARD CATEGORY
            # decided to create 3 categories: gold for regular gold cards, icons_heroes for icons and heroes cards, special for all other special cards
            rarity_link = soup.select_one("a[href*='players?version=']")
            if rarity_link:
                rarity_span = rarity_link.select_one("span.text-ellipsis")
                if rarity_span:
                    rarity_text = get_text(rarity_span)
                    if rarity_text and "Gold" in rarity_text:
                        row["card_category"] = "Gold"
                    elif rarity_text and ("Icon" in rarity_text or rarity_text == "Heroes"):
                        row["card_category"] = "Icons_Heroes"
                    else:
                        row["card_category"] = "Special"
                else:
                    row["card_category"] = None
            else:
                row["card_category"] = None

            # EXTRACT STATS (PACE, SHOOTING, PASSING, DRIBBLING, DEFENDING, PHYSICAL)
            stats_divs = card.select("div.playercard-26-stats.playercard-stats")
            for div in stats_divs:
                label = get_text(div.select_one("span.playercard-26-stat-value"))
                value = get_text(div.select_one("div.playercard-26-stat-number"))
                if label and value:
                    mapping = {
                        "Pac": "pace",
                        "Sho": "shooting",
                        "Pas": "passing",
                        "Dri": "dribbling",
                        "Def": "defending",
                        "Phy": "physical",
                    }
                    if label in mapping:
                        row[mapping[label]] = int(value)

            # EXTRACT SKILL MOVES
            skills_label = soup.find("div", class_="xs-font uppercase text-faded", string="Skills")
            if skills_label:
                parent_row = skills_label.parent
                value_div = parent_row.select_one("div.xxs-row.align-center")
                if value_div:
                    value_text = value_div.get_text(strip=True)
                    row["skill_moves"] = int(value_text[0])

            # EXTRACT WEAK FOOT
            wf_label = soup.find("div", class_="xs-font uppercase text-faded", string="Weak Foot")
            if wf_label:
                parent_row = wf_label.parent
                value_div = parent_row.select_one("div.xxs-row.align-center")
                if value_div:
                    value_text = value_div.get_text(strip=True)
                    row["weak_foot"] = int(value_text[0])

            # EXTRACT NUMBER OF PLAYSTYLES AND PLAYSTYLES+
            abilities_block = None
            for div in soup.select("div.player-abilities-wrapper.xs-column"):
                classes = div.get("class", [])
                if "hidden" not in classes:
                    abilities_block = div
                    break

            if abilities_block:
                links = abilities_block.select("a.playStyle-table-icon")
                #extract number of playstyles
                row["num_playstyles"] = sum(
                    "psplus" not in (a.get("class") or []) for a in links
                )
                #extract number of playstyles+
                row["num_playstyles_plus"] = sum(
                    "psplus" in (a.get("class") or []) for a in links
                )
            else:
                row["num_playstyles"] = 0
                row["num_playstyles_plus"] = 0

            # EXTRACT NATIONALITY, LEAGUE AND CLUB
            # heroes and icons have heroes/icons as club
            info_row = card.select_one("div.playercard-26-info-row")
            if info_row:
                imgs = info_row.select("img")
                for img in imgs:
                    title = img.get("title")
                    src = img.get("src", "")
                    if "/nation/" in src:
                        row["nationality"] = title
                    elif "/league/" in src:
                        row["league"] = title
                    elif "/clubs/" in src:
                        row["club"] = title

            # EXTRACT NUMBER OF PLAYABLE POSITIONS AND POSITIONS
            positions = []
            # all possible playable positions in the game
            valid_pos = {"LB", "LWB", "CB", "RB", "RWB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST"}

            #extract main position
            main_pos_el = card.select_one("div.playercard-26-position")
            if main_pos_el:
                main_pos = get_text(main_pos_el)
                if main_pos in valid_pos:
                    positions.append(main_pos)

            #extract alternative positions
            alt_pos_els = card.select("div.playercard-26-alt-pos-sub")
            for alt_el in alt_pos_els:
                alt_pos = get_text(alt_el)
                if alt_pos in valid_pos:
                    positions.append(alt_pos)

            row["positions"] = ",".join(positions) # shows all the playable positions each player can play
            row["num_positions"] = len(positions) # shows number of playable positions for each player

            # EXTRACT POSITION CLUSTERS
            # center backs are less expensive than strikers, but some position tend to have similar prices, so I created position clusters to capture that
            # clusters: centerbacks (CB), fullbacks (LB, LWB, RB, RWB), midfielders (CDM, CM), attacking midfielders (CAM, LM, LW, RM, RW) and strikers (ST)
            pos_set = set(positions)

            def in_any(possible):
                """ Checks if player can play in specific cluster (1) or not (0) """
                return any(p in pos_set for p in possible)

            row["cluster_cb"] = 1 if in_any(["CB"]) else 0
            row["cluster_fullbacks"] = 1 if in_any(["LB", "LWB", "RB", "RWB"]) else 0
            row["cluster_mid"] = 1 if in_any(["CDM", "CM"]) else 0
            row["cluster_att_mid"] = 1 if in_any(["CAM", "LM", "LW", "RM", "RW"]) else 0
            row["cluster_st"] = 1 if in_any(["ST"]) else 0

            rows.append(row)
            time.sleep(1) # Avoid too many requests in a short time

        # Acknowledge if there was an error with a specific player's attributes
        except requests.exceptions.RequestException as e:
            logger.error(f"Error on player {index}: {e}")
            row = {"url": url}
            rows.append(row)
            failed_players.append(index)
            continue

    # Acknowledge the number of player attributes that failed to scrape
    if failed_players:
        logger.warning(f"{len(failed_players)} players failed to scrape")

    # Save scraped attributes to CSV file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    output_path = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
    df.to_csv(output_path, index=False)

    logger.info(f"Scraped {len(df)} player attributes to {output_path}")
