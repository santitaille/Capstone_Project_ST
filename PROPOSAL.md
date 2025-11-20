# Project Proposal - Santiago Tailleferd

## Title: Explaining and Predicting the Market Value of EA FC 26 Player Cards (83+ OVR)

### Motivation
Player prices in EA FC 26 vary widely: the cheapest player costs 200 credits, while the most expensive player costs 11.49M credits. That wide range can be explained by many variables, including overall rating (OVR), nationality, league, rarity and other ongoing market dynamics. Prices evolve rapidly due to promo releases and SBC requirements, making it hard to determine which characteristics genuinely drive value. This project aims to explain which characteristics most strongly influence prices and to predict player prices using machine learning.

### Planned approach and technologies
I will scrape two snapshots of all 83+ OVR tradeable non-goalkeeper cards from Futbin 1 week apart. Then, I will clean and merge the data (for example, keeping only players that are present in both scrapes). The dataset will include card type (gold, special, hero/icon), card identity (gender, nationality, league, club), card attributes (pace, shooting, passing, dribbling, defending, physical, weak foot, skills move, number of playstyles and number of playstyles+) and position features: number of playable positions of the player and zone indicators that capture all positional areas a player can cover: center backs (CB), fullbacks (LB, LWB, RB, RWB), midfielders (CDM, CM), attacking midfielders (CAM, LM, LW, RM, RW) and strikers (ST).

Then, I will train 4 models: linear regression, random forest, XGBoost and a neural network (MLP) and compare them against 2 baselines: median price per OVR rating and median price per (OVR rating x card type). Evaluation will use temporal validation: the model will be trained on the first scrape and tested on the prices of the second scrape.

### Expected challenges
There are more than 15’000 cards in the game, however, only 83+ OVR rated players are really used in the game (around 1’000 players). This sample will make the analysis more meaningful. Another challenge is volatility in EA FC 26 market. However, temporal validation will mitigate it by forcing the model to generalize to future prices and not overfitting the data from the first scrape. Heterogeneity between gold, special and icon/hero cards will be handled through identity variables. Finally, nonlinear relations will be captured with the random forest and XGBoost.

### Success criteria
This project will be considered successful if it outperforms both baselines on the second scrape prices. Also, the model will have to produce logical trends (for example, a 95 OVR player with high pace should cost more than an 83 OVR with less pace).

### Stretch goals
If time permits, I would like to use SHAP values to identify which attributes contribute the most strongly to the model predictions. Secondly, I would like to detect potentially underpriced/overpriced players by comparing their actual price against their predicted price.