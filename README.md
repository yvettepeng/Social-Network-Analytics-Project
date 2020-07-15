# Social-Network-Analytics-Project
This project uses the transactional data of a mobile payment app and predicts a user's transactions by employing the classic RF (Recency, Frequency) model and user's social network metrics. The dataset is not provided for non-disclosure reason. The code for the project uses packages including PySpark and NetworkX and consists of three parts:
## 1. Text & Emoji Analytics
This part classifies users' transactions by analyzing the text and emoji used in the transaction description. Then, the dynamic spending profiles of users will be calculated for each month over his or her lifetime (0-12 months). For example, assume a user’s first transaction was a pizza emoji. Then, her user profile at lifetime point 0 would be 100% food. Now, by the end of her first month in Venmo, she has transacted 4 times, 2 of them are food and 2 are activity related. Her spending profile in lifetime point, 1 month, is 50% food and 50% activity. The spending profile is dynamic as it will evolve over the user's lifetime.
## 2. Social Network Analytics
This part calculates users' social network metrics e.g. number of friends, clustering coefficients and page rank. These metrics are also dynamic, and are computed for each month over the users' lifetime (0-12 months). 
## 3. Predictive Analytics 
This part explores how spending profile and social network metrics calculated at each lifetime point (0-12 months) would predict **Y** - the total number of transactions during the user's first year's lifetime. We use 4 models and compare their MSE to see if social network metrics can raise the predictive power.
#### Model 1: 
For each user’s lifetime point, regress recency and frequency on Y. Recency refers to the last time a user was active, and frequency is how often a user uses the mobile payment app in a month. <br />
The MSE of Model 1 is plotted as below: <br />
![image](https://github.com/yvettepeng/Social-Network-Analytics-Project/blob/master/MSE_Model%201.png)
#### Model 2: 
For each user’s lifetime point, regress recency, frequency and spending profile on Y.
The MSE of Model 2 is plotted as below:
#### Model 3: 
For each user’s lifetime point, regress social network metrics on Y.
The MSE of Model 3 is plotted as below:
#### Model 4: 
For each user’s lifetime point, regress social network metrics and spending profile of her social network (i.e. friends and friends of friends) on Y.
The MSE of Model 4 is plotted as below:
