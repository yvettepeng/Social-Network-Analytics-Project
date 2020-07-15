# Social-Network-Analytics-Project
This project uses the transactional data of a mobile payment app and predicts a user's spending behavior by employing the classic RF (Recency, Frequency) model and user's social network metrics. The dataset is not provided for non-disclosure reason. The code for the project uses packages including PySpark and NetworkX and consists of three parts:
## 1. Text & Emoji Analytics
This part classifies users' transactions by analyzing the text and emoji used in the transaction description. Then, the dynamic spending profiles of users will be calculated for each month over his or her lifetime. For example, assume a user’s first transaction was a pizza emoji. Then, her user profile at 0
would be 100% food. Now, by the end of her first month in Venmo, she has transacted 4 times, 2 of them are food and 2 are activity related. Her speding profile in 1 month is 50% food and 50% activity.
## 2. Social Network Analytics
This part calculates users' social network metrics e.g. number of friends, clustering coefficients and page rank. These metrics are computed dynamically for each month over the users' lifetime. 
