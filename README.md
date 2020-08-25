# Kaggle_House_Prices_Advanced_Regression_Techniques_0.11701_top10.6_only_one_entry
Kaggle House Prices: Advanced Regression Techniques. Public Leaderboard Score 0.11701 with ONLY ONE ENTRY.



## Preface
I used to rank top5% on [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

![Public Score with Multiple Entries](https://github.com/mliw/Kaggle_House_Prices_Advanced_Regression_Techniques_0.11701_top10.6_percent_only_one_entry/blob/master/pics/0.PNG)

However, more than 100 entries were made to achieve this score, which means I'm actually tuning my model on the test data set. The problem of data leakage arises when you make submissions more than once.

This repo aims to tell you how to get a score of 0.11701(top10.57%) with only **ONE ENTRY**. With the help of Genetic Algorithm and Bayesian Optimization(hyperopt), this is not a lucky score achieved randomly, but a certain result we can **ALMOST SURELY** get. The document of this repo is at [here](https://github.com/mliw/Kaggle_House_Prices_Advanced_Regression_Techniques_0.11701_top10.6_percent_only_one_entry/blob/master/doc/Tutorial.pdf).

![Public Score with One Entries](https://github.com/mliw/Kaggle_House_Prices_Advanced_Regression_Techniques_0.11701_top10.6_percent_only_one_entry/blob/master/pics/1.PNG)

## Single Models
8 base models are involved in this repo, and the optimization of these models is involved in files like 1_single_model_svr_2.py. All files start with "1_" conduct Genetic Algorithm and Bayesian Optimization(hyperopt) independently.

## Model Stacking
Greedy algorithm is involved in model stacking(2_stacking.py).

## Result
Results of stacking are gathered at [here](https://github.com/mliw/Kaggle_House_Prices_Advanced_Regression_Techniques_0.11701_top10.6_percent_only_one_entry/tree/master/stacking).
The final submission is final_5_0.10117320406891336.csv



