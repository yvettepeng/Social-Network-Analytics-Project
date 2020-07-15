#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This script employs pyspark as the main tool to analyze the spending behavior and the social network of Venmo users using their transactional data. The script consists of 5 parts:
# 1. Initial set-up <br/>
# 2. Import datasets <br/>
# 3. Part 1 Text Analytics <br/>
# 4. Part 2 Social Network Analytics <br/>
# 5. Part 3 Predictive Analytics with MLlib <br/>
# 
# As the size of the dataset is large, we suggest users of the script running on google colab or clusters. It is also recommended that users save the regression input tables of Part 3 as parquet files and then estimate predictive models, as parquet files will dramatically improve the speed of running the regression.

# # Set up

# In[ ]:


get_ipython().system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
get_ipython().system('wget -q https://www-us.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz')
get_ipython().system('tar xf spark-2.4.5-bin-hadoop2.7.tgz')
get_ipython().system('pip install -q findspark')


# In[ ]:


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-1.8.0-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.5-bin-hadoop2.7"


# In[ ]:


import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.config("spark.driver.memory", "10g").appName('Homework2').getOrCreate()


# In[ ]:


get_ipython().system('pip install emoji')
get_ipython().system('pip install pyspark')


# In[ ]:


import emoji
from emoji import *
import string
import re
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext, Window
from pyspark.sql.functions import lit, sum, col, min, when, pandas_udf, PandasUDFType, regexp_extract
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType
from pyspark.conf import SparkConf
SparkSession.builder.config(conf=SparkConf())
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import networkx as nx
from itertools import groupby
from collections import OrderedDict
import matplotlib.pyplot as plt


# # Import Datasets

# In[ ]:


# this helps you connect to the google drive
from google.colab import drive
drive.mount('/content/drive',force_remount=True)


# 

# In[ ]:


get_ipython().system("ls '/content/drive/My Drive/ConFiveDance/code/VenmoSample.snappy.parquet'")


# In[ ]:


inputdata = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/VenmoSample.snappy.parquet')


# In[ ]:


##Import text dictionary
text_dict = spark.read.csv('/content/drive/My Drive/ConFiveDance/code/Venmo_Word_Classification_Dictonary.csv', header=True)


# In[ ]:


##Import emoji dictionary
emoji_dict = spark.read.csv('/content/drive/My Drive/ConFiveDance/code/Venmo_Emoji_Classification_Dictionary.csv', header=True)


# #Part 1 Text Analytics

# ## Q1 Classify Venmo’s transactions

# ### Dictionaries Preparation

# Text Dictionary

# In[ ]:


##make text dictionary lists based on categories
txt_lst_Event=[row['Event'] for row in text_dict.collect()]
txt_lst_Event=[i for i in txt_lst_Event if i!= None]
txt_lst_Travel=[row['Travel'] for row in text_dict.collect()]
txt_lst_Travel=[i for i in txt_lst_Travel if i!= None]
txt_lst_Food=[row['Food'] for row in text_dict.collect()]
txt_lst_Food=[i for i in txt_lst_Food if i!= None]
txt_lst_Activity=[row['Activity'] for row in text_dict.collect()]
txt_lst_Activity=[i for i in txt_lst_Activity if i!= None]
txt_lst_Transportation=[row['Transportation'] for row in text_dict.collect()]
txt_lst_Transportation=[i for i in txt_lst_Transportation if i!= None]
txt_lst_People=[row['People'] for row in text_dict.collect()]
txt_lst_People=[i for i in txt_lst_People if i!= None]
txt_lst_Utility=[row['Utility'] for row in text_dict.collect()]
txt_lst_Utility=[i for i in txt_lst_Utility if i!= None]
txt_lst_Cash=[row['Cash'] for row in text_dict.collect()]
txt_lst_Cash=[i for i in txt_lst_Cash if i!= None]
txt_lst_Illegal_Sarcasm=[row['Illegal/Sarcasm'] for row in text_dict.collect()]
txt_lst_Illegal_Sarcasm=[i for i in txt_lst_Illegal_Sarcasm if i!= None]


# In[ ]:


##Create a list that contains all words that can be found in the text dictionary
txt_lst = txt_lst_Event+txt_lst_Travel+txt_lst_Food+txt_lst_Activity+txt_lst_Transportation+txt_lst_People+txt_lst_Utility+txt_lst_Cash+txt_lst_Illegal_Sarcasm


# Emoji Dictionary

# In[ ]:


##Clean data: remove messy symbols
emoji_pd=emoji_dict.toPandas()
emoji_pd = emoji_pd.applymap(lambda x:x[0] if x is not None else x)
emoji_spark=spark.createDataFrame(emoji_pd)


# In[ ]:


##make emoji dictionary lists based on categories
emoji_lst_Event=[row['Event'] for row in emoji_spark.collect()]
emoji_lst_Event=[i for i in emoji_lst_Event if i!= None]
emoji_lst_Travel=[row['Travel'] for row in emoji_spark.collect()]
emoji_lst_Travel=[i for i in emoji_lst_Travel if i!= None]
emoji_lst_Food=[row['Food'] for row in emoji_spark.collect()]
emoji_lst_Food=[i for i in emoji_lst_Food if i!= None]
emoji_lst_Activity=[row['Activity'] for row in emoji_spark.collect()]
emoji_lst_Activity=[i for i in emoji_lst_Activity if i!= None]
emoji_lst_Transportation=[row['Transportation'] for row in emoji_spark.collect()]
emoji_lst_Transportation=[i for i in emoji_lst_Transportation if i!= None]
emoji_lst_People=[row['People'] for row in emoji_spark.collect()]
emoji_lst_People=[i for i in emoji_lst_People if i!= None]
emoji_lst_Utility=[row['Utility'] for row in emoji_spark.collect()]
emoji_lst_Utility=[i for i in emoji_lst_Utility if i!= None]


# In[ ]:


##Create a list that contains all emojis that can be found in the emoji dictionary
emoji_lst=emoji_lst_Event+emoji_lst_Travel+emoji_lst_Food+emoji_lst_Activity+emoji_lst_Transportation+emoji_lst_People+emoji_lst_Utility


# Functions

# In[ ]:


##A function that takes a word or a emoji as input and returns the category of the input
def to_category(wd):
    if wd in txt_lst_Event or wd in emoji_lst_Event:
        return 'Event'
    elif wd in txt_lst_Travel or wd in emoji_lst_Travel:
        return 'Travel'
    elif wd in txt_lst_Food or wd in emoji_lst_Food:
        return 'Food'
    elif wd in txt_lst_Activity or wd in emoji_lst_Activity:
        return 'Activity'
    elif wd in txt_lst_Transportation or wd in emoji_lst_Transportation:
        return 'Transportation'
    elif wd in txt_lst_People or wd in emoji_lst_People:
        return 'People'
    elif wd in txt_lst_Utility or wd in emoji_lst_Utility:
        return 'Utility'
    elif wd in txt_lst_Cash:
        return 'Cash'
    elif wd in txt_lst_Illegal_Sarcasm:
        return 'Illegal_Sarcasm'
    else:
        return 'Not_classified'


# In[ ]:


##A funtion that takes strings as input and returns the category of the string
##If a single string involves more than one category, the returned category should be the one that has the greatest occurrance
def classify(string):

    if bool(emoji.get_emoji_regexp().search(string)) is True:
    #case when the string contains emoji
        #create a list that includes all the emojis in the string
        emoji_in_str=list(filter(lambda x:  x in string, emoji_lst))
    else:
        #if the string does not contain any emoji, create an empty list
        emoji_in_str=[]

    #letters to lower case
    new_str=string.lower()
    #find all numbers in the string and replace with a space
    new_str=re.sub(r'[0-9]+', ' ', new_str)
    #find all special symbols in the string and replace with a space
    new_str=re.sub('[^A-Za-z0-9]+', ' ', new_str)
    #find all emojis in the string and replace with a space
    no_emoji=re.sub('\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff]+',' ',new_str)
    #replace all >1 spaces with a space
    no_emoji=re.sub('\s+',' ',no_emoji)
    #split the text only string by space into a list of word(s)
    txt_only_lst=no_emoji.split()
    ##filter the list of words and only keep the words which occur in the text dictionary
    txt_in_list=list(filter(lambda x:  x in txt_only_lst, txt_lst))
    #combine the list with emojis and the list with words into a new list
    comb_lst=txt_in_list+emoji_in_str

    if comb_lst==[]:
    #when none of the emojis nor words has a match in category, the string has no category
        cat='Not_classified'
    else:
        #create a new list that lists the categories of the elements in the string
        cat_lst=list(map(to_category, comb_lst))
        #find the category that has the greatest occurrance in the string
        cat=max(cat_lst, key=cat_lst.count)

    return cat


# In[ ]:


import pyspark.sql.functions as F
from pyspark.sql.types import *
udf_classify=F.udf(classify, StringType())


# In[ ]:


#add a new column to the original dataset classifying the category of description for each transaction
input_with_cat=inputdata.withColumn('category',udf_classify('description'))
input_with_cat.show(5)


# ## Q2 

# ### Emoji Only Percentage

# In[ ]:


##the function takes in a string, return 1 if it is a emoji-only string, and 0 otherwise
def emoji_only(desc):
    if all(ele in emoji.UNICODE_EMOJI for ele in desc) is True:
        return 1
    else:
        return 0


# In[ ]:


##add a new column to the original table labeling if the description of transaction is emoji only
udf_emoji_only=F.udf(emoji_only, StringType())
with_label=inputdata.withColumn('Emoji_Only',udf_emoji_only('description'))
with_label.show(5)


# In[ ]:


from pyspark import SparkContext
sc =SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# In[ ]:


##sum up the number of emoji_only transactions
with_label.registerTempTable('with_label_tbl')
table=sqlContext.sql('select sum(Emoji_Only) as sum_emoji_only from with_label_tbl')
table.show()


# In[ ]:


##emoji only percentage
print(1665871.0/inputdata.count())
print("{0:.0%}".format(1665871.0/inputdata.count()))


# ### Top 5 Most Popular Emojis

# In[ ]:


##the function extracts all emojis from the input string
def extract_emojis(string):
    return ''.join(c for c in string if c in emoji.UNICODE_EMOJI)


# In[ ]:


##add a column to the inputdata with the extracted emojis from description
udf_extract_emojis=F.udf(extract_emojis, StringType())
with_emoji_col=inputdata.withColumn('emoji',udf_extract_emojis('description'))


# In[ ]:


##fill the blanks in emoji column with null values
with_emoji_col = with_emoji_col.withColumn('emoji', when(col('emoji') == '', None).otherwise(col('emoji')))
with_emoji_col.show(5)


# In[ ]:


##filter the rows and only keep the transactions which are emoji_only
only_emoji=with_emoji_col.filter(with_emoji_col.emoji.isNotNull())
only_emoji.show(5)


# In[ ]:


##extract all the emojis from emoji column and put them into a list
##each element in the list is a single emoji
emoji_lst=[row.emoji for row in only_emoji.collect()]
emoji_lst=''.join(emoji_lst)
lst_all_emoji=[]
for item in emoji_lst:
    lst_all_emoji.append(item)


# In[ ]:


##create a dictionary to list out the occurrances of each emoji
emoji_num = {value: len(list(freq)) for value, freq in groupby(sorted(lst_all_emoji))}


# In[ ]:


##sort the dictionary by occurrances of emojis in descending order
emoji_sorted = sorted(emoji_num, key=emoji_num.get, reverse=True)
order=0
for i in emoji_sorted:
    print(i, emoji_num[i])
    order+=1
    if order == 5: break


# ### Top 3 most popular emoji categories

# In[ ]:


##create a for loop to count the occurrences of emojis in each category
count_Event=0
count_Travel=0
count_Food=0
count_Activity=0
count_Transportation=0
count_People=0
count_Utility=0
count_Not_classified=0

for item in emoji_lst:
    if item in emoji_lst_Event:
        count_Event=count_Event+1
    elif item in emoji_lst_Travel:
        count_Travel=count_Travel+1
    elif item in emoji_lst_Food:
        count_Food=count_Food+1
    elif item in emoji_lst_Activity:
        count_Activity=count_Activity+1
    elif item in emoji_lst_Transportation:
        count_Transportation=count_Transportation+1
    elif item in emoji_lst_People:
        count_People=count_People+1
    elif item in emoji_lst_Utility:
        count_Utility=count_Utility+1
    else:
        count_Not_classified=count_Not_classified+1


# In[ ]:


##list the categories along with the occurrances
d={'Category': ['Event', 'Travel', 'Food', 'Activity', 'Transportation', 'People', 'Utility'],'Count': [count_Event, count_Travel, count_Food, count_Activity, count_Transportation, count_People, count_Utility]}
cnt_cat = pd.DataFrame(data=d)
cnt_cat.sort_values(by='Count', ascending=False)


# ## Q3 Spending Profile

# In[ ]:


input_with_cat.registerTempTable('input_with_cat')


# In[ ]:


##create a table to list the users, categories, transaction(s) per category and percentage of each category for each user
trans_profile=spark.sql("select user1, Category, count(*) as transaction_per_category, round(count(*)/sum(count(*)) over(partition by user1), 2) as ratio from input_with_cat where transaction_type='payment' group by user1, Category order by user1")


# In[ ]:


trans_profile.show()


# In[ ]:


#pivot the table to create a spending profile for each user
profile=trans_profile.groupby(trans_profile.user1).pivot("Category").avg("ratio")


# In[ ]:


profile.show()


# ## Q4

# In[ ]:


#use window function to get the date of the first transaction of each user1
window = Window.partitionBy("user1").orderBy("datetime") 

dynamic_spending = input_with_cat.select('user1', 'category', 'datetime').withColumn("start_date", min("datetime").over(window))
#dynamic_spending.show()


# In[ ]:


dynamic_spending.createOrReplaceTempView("table_ds1")
dynamic_spending_1 = spark.sql("""SELECT user1, category, datetime, start_date, 
                           ceil(datediff(datetime, start_date)/30) AS lifetime
                           FROM table_ds1
                           ORDER BY user1, datetime""")
#dynamic_spending_1.show(10)


# In[ ]:


# select transaction within one year 
dynamic_spending_1.createOrReplaceTempView("table_ds2")
user_lifetime.createOrReplaceTempView("table_ul1")
dynamic_spending_one_year1 = spark.sql('''
                                      SELECT user1, Category, lifetime,  \
                                            count(*) as transaction_per_category_per_lifetime,  \
                                            round(count(*)/sum(count(*)) over(partition by user1,lifetime), 2) as ratio  
                                      FROM
                                      (SELECT user1, category, lifetime
                                      FROM table_ds2
                                      WHERE lifetime <= 12)
                                      group by user1, lifetime, Category 
                                      order by user1, lifetime
                                      ''')

#dynamic_spending_one_year1.show(15)


# In[ ]:


# Pivot the tables group by user1 and lifetime
categories = sorted(dynamic_spending_one_year1.select("Category").distinct().rdd.map(lambda row: row[0]).collect())

cols1 = [when(col("Category") == cats, col("ratio")).otherwise(None).alias(cats) for cats in categories]

maxs1 = [F.max(col(cats)).alias(cats) for cats in categories]

dynamic_profile = (dynamic_spending_one_year1.select(col("user1"),col("lifetime"), *cols1)                   .groupBy("user1", "lifetime").agg(*maxs1).na.fill(0)                   .orderBy("user1", "lifetime")
                   )
dynamic_profile.createOrReplaceTempView("table_dp")
#dynamic_profile.show(10)


# In[ ]:


# Store the table
dynamic_profile.coalesce(1).write.format("parquet").mode("append").save("dynamic_profile.parquet") 
get_ipython().system('mv dynamic_profile.parquet /content/drive/My\\ Drive/ConFiveDance/code     ')


# # Part 2 Social Network Analysis

# ## Q5

# In[ ]:


inputdata.show(5)


# In[ ]:


venmo = inputdata.rdd


# In[ ]:


# union all the transaction pairs with a list reversing the transactions
user_list = venmo.map(lambda row: (row[0], row[1])).distinct().union(venmo.map(lambda row: (row[1], row[0])).distinct())


# In[ ]:


# reduce to key, [values] pairs, i.e. find friends
find_friends = user_list.groupByKey()


# In[ ]:


find_friends.mapValues(list).take(10)


# In[ ]:


find_friends_as_map = find_friends.mapValues(list).collectAsMap()


# In[ ]:


fof = find_friends.map(lambda row : fndoffnd_func(row, find_friends_as_map))  

def fndoffnd_func(row, rdd):
    res = set()
    for key in row[1]:
        set_bin = set(rdd[key])
        res.update(set_bin)
    return (row[0], list(res))


# In[ ]:


fof.take(2)


# In[ ]:


fof_2d_only = fof.map(lambda row: (row[0], set(row[1]).difference(set(find_friends_as_map[row[0]]))))


# In[ ]:


fof_2d_only.take(2)


# ## Q6

# ### i)

# #### Number of friends

# In[ ]:


dynamic_social = inputdata.select('user1', 'user2', 'datetime').union(inputdata.select('user2', 'user1', 'datetime'))


# In[ ]:


# use window function to get the date of the first transaction of each user1
window = Window.partitionBy("user1").orderBy("datetime") 
dynamic_social = dynamic_social.withColumn("start_date", min("datetime").over(window))
dynamic_social.show(50)


# In[ ]:


dynamic_social.createOrReplaceTempView("table1")
# calculate the lifetime month of the user1 for each line of transaction
dynamic_social = spark.sql("""SELECT user1, user2, datetime, start_date, 
                           ceil(datediff(datetime, start_date)/30) AS lifetime
                           FROM table1 
                           ORDER BY user1, datetime""")
dynamic_social.show(50)


# In[ ]:


# since we only count the cumulative number of "new" friends met in each lifetime
# get the first 'lifetime' value when each pair of users met
dynamic_social.createOrReplaceTempView("table1")
num_friends = spark.sql("""SELECT user1, user2, MIN(lifetime) AS lifetime
                           FROM table1 
                           GROUP BY user1, user2
                           ORDER BY user1, MIN(lifetime)""")
friend_list = num_friends
num_friends.show(10)


# In[ ]:


# use map reduce to create a dataframe with two columns: userid and lifetime that is consecutive from 0 to 12 months
user_lifetime = dynamic_social.select("user1", "lifetime").rdd
user_lifetime = user_lifetime.flatMapValues(lambda value:range(0,13))
# convert user_lifetime to a dataframe and rename the columns
user_lifetime = user_lifetime.toDF(["user1", "lifetime"])
user_lifetime.show(20)


# In[ ]:


# change the schema of user_lifetime, make two columns as integer type
user_lifetime = user_lifetime.select(user_lifetime.user1.cast(IntegerType()), 
                                     user_lifetime.lifetime.cast(IntegerType()))
user_lifetime.printSchema()


# In[ ]:


num_friends.createOrReplaceTempView("num_friends")
user_lifetime.createOrReplaceTempView("user_lifetime")
num_friends_one_year = spark.sql("""
                        SELECT user1, lifetime, COUNT(DISTINCT user2) AS num_friends
                        FROM 
                          (SELECT user1, lifetime, user2 FROM num_friends
                          UNION
                          SELECT user1, lifetime, null AS user2 FROM user_lifetime) union_table
                        WHERE lifetime <= 12
                        GROUP BY user1, lifetime
                        ORDER BY user1, lifetime
                        """)
num_friends_one_year.show(20)


# In[ ]:


# calculate the cumulative number of friends in each lifetime month for each user
num_fnd_window = Window.partitionBy("user1").orderBy("lifetime") 
num_friends_one_year = num_friends_one_year.withColumn("cum_num_friends", 
                                                       sum("num_friends").over(num_fnd_window)).sort(col("user1"), 
                                                                                                     col("lifetime"))
num_friends_one_year.show(15)


# #### Number of friends of friends

# In[ ]:


dynamic_social.createOrReplaceTempView("table1")
dynamic_social.createOrReplaceTempView("table2")
# self join the dynamic_social table, and select the first lifetime month that user met the friend of friend
dynamic_social_self_join = spark.sql("""SELECT table1.user1, table2.user2 as friend_of_friend,
                                    MIN(table1.lifetime) AS lifetime
                                    FROM table1
                                    LEFT JOIN table2 ON table1.user2 = table2.user1
                                    AND table1.user1 != table2.user2
                                    AND table1.datetime >= table2.datetime
                                    WHERE table1.lifetime <= 12
                                    GROUP BY table1.user1, table2.user2
                                    ORDER BY table1.user1, MIN(table1.lifetime)""")
dynamic_social_self_join.show(15)


# In[ ]:


dynamic_social_self_join.createOrReplaceTempView("table1")
friend_list.createOrReplaceTempView("table2")
# only consider the friend of friend whom user1 doesn't know. exclude the 1st degree friend
num_friends_of_friends = spark.sql("""SELECT table1.user1, friend_of_friend, table1.lifetime
                                   FROM table1
                                   LEFT JOIN table2 ON table1.user1 = table2.user1
                                   AND table1.lifetime = table2.lifetime
                                   AND table1.friend_of_friend = table2.user2
                                   WHERE table2.user2 is null
                                   OR friend_of_friend is null
                                   ORDER BY table1.user1, table1.lifetime
                                    """)
friends_of_friends = num_friends_of_friends
num_friends_of_friends.show(10)


# In[ ]:


num_friends_of_friends.createOrReplaceTempView("num_friends_of_friends")
user_lifetime.createOrReplaceTempView("user_lifetime")
# make a consecutive 0-12 month lifetime
num_friends_of_friends_one_year = spark.sql("""
                        SELECT user1, lifetime, COUNT(DISTINCT friend_of_friend) AS num_fnd_of_fnd
                        FROM 
                          (SELECT user1, lifetime, friend_of_friend FROM num_friends_of_friends
                          UNION
                          SELECT user1, lifetime, null AS friend_of_friend FROM user_lifetime) union_table
                        WHERE lifetime <= 12
                        GROUP BY user1, lifetime
                        ORDER BY user1, lifetime
                        """)
num_friends_of_friends_one_year.show(20)


# In[ ]:


# use window function to calculate the cumulative number of "new" friends of friends for each lifetime month
num_fnd_fnd_window = Window.partitionBy("user1").orderBy("lifetime") 
num_friends_of_friends_one_year = num_friends_of_friends_one_year.withColumn("cum_num_fnd_of_fnd", 
                                sum("num_fnd_of_fnd").over(num_fnd_fnd_window)).sort(col("user1"), 
                                                                                           col("lifetime"))
num_friends_of_friends_one_year.show(20)


# ### ii) Cluster Coefficient

# In[ ]:


num_friends_one_year.createOrReplaceTempView("num_friends")
# calculate the number of possible vertices for each lifetime month
num_vertices = spark.sql("""SELECT user1, lifetime, cum_num_friends*(cum_num_friends-1)/2 AS num_vertices
                            FROM num_friends
                            ORDER BY user1, lifetime""")
num_vertices.show(15)


# Or use rdd and map function to generate # possible vertices.

# In[ ]:


# num_friends_rdd = num_friends_one_year.select("user1", "lifetime", "cum_num_friends").rdd
# num_vertices_rdd = num_friends_rdd.map(lambda row: num_vertices(row))

# def num_vertices(row):
#   vertice = row[2]*(row[2]-1)/2
#   return(row[0], row[1], vertice)
# num_vertices_df = num_vertices_rdd.toDF(["user1", "lifetime", "num_vertices"])
# num_vertices_df = num_vertices_df.select(num_vertices_df.user1.cast(IntegerType()), 
#                                      num_vertices_df.lifetime.cast(IntegerType()),
#                                      num_vertices_df.num_vertices.cast(IntegerType()))
# num_vertices_df.printSchema()
# num_vertices_df.show(5)


# In[ ]:


dynamic_social.createOrReplaceTempView("t1")
# only consider the date time when two users first transacted
edges = spark.sql("""SELECT user1, user2, MIN(datetime) AS datetime, MIN(lifetime) AS lifetime, MIN(start_date) AS start_date
                      FROM t1 
                      GROUP BY user1, user2
                      ORDER BY user1, MIN(datetime)""")
edges.show(5)


# In[ ]:


edges.createOrReplaceTempView("t1")
edges.createOrReplaceTempView("t2")
edges.createOrReplaceTempView("t3")
# self join the edges table for three times, and make user2 of the table 3 equal to the user1 of the table1
edges_3join = spark.sql("""SELECT t1.user1 AS t1_user1, t1.user2 AS t1_user2, 
                        t1.start_date AS t1_start_date, t1.datetime AS t1_datetime, 
                        t2.user1 AS t2_user1, t2.user2 AS t2_user2, t2.datetime AS t2_datetime, 
                        t3.user1 AS t3_user1, t3.user2 AS t3_user2, t3.datetime AS t3_datetime
                      FROM t1
                      LEFT JOIN t2 ON t1.user2 = t2.user1
                      AND t1.user1 != t2.user2
                      LEFT JOIN t3 ON t2.user2 = t3.user1
                      AND t2.user1 != t3.user2
                      WHERE t1.user1 == t3.user2
                      ORDER BY t1.user1, t1.lifetime""")
edges_3join.show(20)
# for table1 user1, each triangle will appear twice in the results table 


# In[ ]:


edges_3join.createOrReplaceTempView("t1")
# calculate the date when the triangle formed by selecting the latest date of the transaction between three users
# then calculate the lifetime for t1_user1 when the triangle formed 
edges_3join_tridate = spark.sql("""
                                SELECT t1_user1, t2_user1, t3_user1,
                                ceil(DATEDIFF(GREATEST(t1_datetime, t2_datetime, t3_datetime), t1_start_date)/30)
                                AS triangle_lifetime
                                FROM t1
                                """)
edges_3join_tridate.show(5)


# In[ ]:


edges_3join_tridate.createOrReplaceTempView("t1")
# get the triangles of lifetime <= 12, and count the number of triangles group by each user and lifetime
num_triangles = spark.sql("""
                                SELECT t1_user1 AS user1, triangle_lifetime AS lifetime,
                                COUNT(*)/2 AS num_triangles
                                FROM t1
                                WHERE triangle_lifetime <= 12
                                GROUP BY t1_user1, triangle_lifetime
                                ORDER BY t1_user1, triangle_lifetime
                                """)
num_triangles.show(5)


# In[ ]:


num_triangles.createOrReplaceTempView("t1")
user_lifetime.createOrReplaceTempView("t2")
# get # triangles for consecutive 0-12 months
num_triangles_one_year = spark.sql("""
                                SELECT user1, lifetime, SUM(num_triangles) AS num_triangles
                                FROM(
                                  SELECT user1, lifetime, num_triangles
                                  FROM t1
                                  UNION
                                  SELECT user1, lifetime, 0 AS num_triangles
                                  FROM t2) temp_table
                                GROUP BY user1, lifetime
                                ORDER BY user1, lifetime
                                """)
num_triangles_one_year.show(5)


# In[ ]:


# calculate the cumulative # triangles in each lifetime month for each user
num_tri_window = Window.partitionBy("user1").orderBy("lifetime") 
num_triangles_one_year = num_triangles_one_year.withColumn("cum_num_triangles", 
                                                       sum("num_triangles").over(num_tri_window)).sort(col("user1"), 
                                                                                                     col("lifetime"))
num_triangles_one_year.show(15)


# In[ ]:


num_triangles_one_year.createOrReplaceTempView("t1")
num_vertices.createOrReplaceTempView("t2")

cluster_coef = spark.sql("""
                          SELECT t1.user1, t1.lifetime, 
                          cum_num_triangles/NULLIF(num_vertices,0) AS cluster_coefficient
                          FROM t1
                          JOIN t2 ON t1.user1 = t2.user1
                          AND t1.lifetime = t2.lifetime
                          ORDER BY t1.user1, t1.lifetime
                          """)
cluster_coef.show(15)


# ### iii) PageRank

# In[ ]:


transactions = inputdata.select("user1", "user2")
transactions_rdd = transactions.rdd
transactions_tuples = transactions_rdd.map(tuple)


# In[ ]:


transactions_tuples.take(2)


# In[ ]:


txn_tuples = transactions_tuples.collect()


# In[ ]:


G = nx.Graph()
G.add_edges_from(txn_tuples)
pr = nx.pagerank_scipy(G)


# In[ ]:


pr_list = [(key, pr[key]) for key in pr.keys()]


# In[ ]:


schema = StructType([
    StructField("user1", IntegerType(), True),
    StructField("PageRank", DoubleType(), True)
])
pr_df = spark.createDataFrame(pr_list, schema=schema)


# In[ ]:


pr_df.show(5)


# # Part 3 Predictive Analytics with MLlib

# ## Q7

# In[ ]:


transaction_count = inputdata.select('user1', 'user2', 'datetime')


# In[ ]:


# use window function to get the date of the first transaction of each user1
window = Window.partitionBy("user1").orderBy("datetime") 
transaction_count = transaction_count.withColumn("start_date", min("datetime").over(window))
transaction_count.show(50)


# In[ ]:


transaction_count.createOrReplaceTempView("table7")
# calculate the lifetime month of the user1 for each line of transaction
transaction_count = spark.sql("""SELECT user1,COUNT(*) AS num_transaction
FROM
(SELECT user1, user2, datetime, start_date, 
ceil(datediff(datetime, start_date)/30) AS lifetime
FROM table7) lifetime_table
WHERE lifetime <= 12
GROUP BY user1
ORDER BY user1 """)
transaction_count.show(50)


# ## Q8

# In[ ]:


dynamic_social_8 = inputdata.select('user1', 'user2', 'datetime')


# In[ ]:


# use window function to get the date of the first transaction of each user1
window = Window.partitionBy("user1").orderBy("datetime") 
dynamic_social_8 = dynamic_social_8.withColumn("start_date", min("datetime").over(window))
dynamic_social_8.show(50)


# In[ ]:


dynamic_social_8.createOrReplaceTempView("table1")
# calculate the lifetime month of the user1 for each line of transaction
dynamic_social_8 = spark.sql("""SELECT user1, user2, datetime, start_date, 
                           CEIL(DATEDIFF(datetime, start_date)/30) AS lifetime,
                           CEIL(DATEDIFF(datetime, start_date)/30) * 30 AS lifetime_days,
                           DATEDIFF(datetime, start_date) AS days
                           FROM table1 
                           ORDER BY user1, datetime""")
dynamic_social_8.show(10)


# In[ ]:


dynamic_social_8.createOrReplaceTempView("t1")
# calculate the lifetime month of the user1 for each line of transaction
df_rfmodel = spark.sql("""SELECT user1, lifetime, lifetime_days, max(days) AS latest_txn, COUNT(*) AS num_txn 
                           FROM t1
                           WHERE lifetime <= 12
                           GROUP BY user1, lifetime,lifetime_days
                           ORDER BY user1, lifetime""")
df_rfmodel.show(5)


# In[ ]:


# use map reduce to create a dataframe with two columns: userid and lifetime that is consecutive from 0 to 12 months
user_lifetime_8 = dynamic_social_8.select("user1", "lifetime").rdd
user_lifetime_8 = user_lifetime_8.flatMapValues(lambda value:range(0,13))
# convert user_lifetime to a dataframe and rename the columns
user_lifetime_8 = user_lifetime_8.toDF(["user1", "lifetime"])
user_lifetime_8.show(15)


# In[ ]:


# change the schema of user_lifetime, make two columns as integer type
user_lifetime_8 = user_lifetime_8.select(user_lifetime_8.user1.cast(IntegerType()), 
                                     user_lifetime_8.lifetime.cast(IntegerType()))
user_lifetime_8.printSchema()


# In[ ]:


df_rfmodel.createOrReplaceTempView("df_rfmodel")
user_lifetime_8.createOrReplaceTempView("user_lifetime")
df_rfmodel_full = spark.sql("""SELECT user1, lifetime, lifetime_days, MAX(latest_txn) AS latest_txn, 
                        MAX(num_txn) AS num_txn
                        FROM
                          (SELECT user1, lifetime, lifetime_days, latest_txn, num_txn FROM df_rfmodel
                          UNION
                          SELECT user1, lifetime, lifetime * 30 AS lifetime_days, null AS latest_txn, null AS num_txn FROM user_lifetime) union_table
                        GROUP BY user1, lifetime, lifetime_days
                        ORDER BY user1, lifetime """)
df_rfmodel_full.show(5)


# In[ ]:


# calculate frequency and recency
# frequency: how often a user uses Venmo in a month. It is standardized and equals to (number of transactions/30)
# recency: the last time a user was active
    #if a user has used Venmo twice during her first month in Venmo with the second time being on day x, 
    #then her recency in month 1 is “30-x”
df_rfmodel_full.createOrReplaceTempView("df_rfmodel_full")
rf_model = spark.sql("SELECT user1, lifetime, lifetime_days,                 IFNULL(((sum(num_txn) OVER (PARTITION BY user1 ORDER BY lifetime))/lifetime_days, 0) AS frequency,                 (lifetime_days - MAX(latest_txn)                     OVER(PARTITION BY user1                          ORDER BY lifetime ASC)) AS recency               FROM df_rfmodel_full               ORDER BY user1, lifetime")

rf_model.createOrReplaceTempView("rf_model")
rf_model.show(24)


# ## Q9 
# For each user’s lifetime point, regress recency and frequency on Y. Plot the MSE for each lifetime point. In other words, your x-axis will be lifetime in months (0-12), and your yaxis will be the MSE. (Hint: Don’t forget to split your data into train and test sets).

# In[ ]:


transaction_count.createOrReplaceTempView("transaction_count_table")


# In[ ]:


# join tables to combine user lifetime, frequency, recency and number of total transactions (y)
regression9_input = spark.sql("""SELECT user1, lifetime, frequency, recency, num_transaction
                  FROM rf_model JOIN transaction_count_table USING (user1)
                  ORDER BY user1, lifetime""")
regression9_input.createOrReplaceTempView("regression9_input")
regression9_input.show()


# In[ ]:


regression9_input.createOrReplaceTempView("regression9_input")


# In[ ]:


regression9_input_lifetime0 = spark.sql("select * from regression9_input where lifetime=0 ")
regression9_input_lifetime1 = spark.sql("select * from regression9_input where lifetime=1 ")
regression9_input_lifetime2 = spark.sql("select * from regression9_input where lifetime=2 ")
regression9_input_lifetime3 = spark.sql("select * from regression9_input where lifetime=3 ")
regression9_input_lifetime4 = spark.sql("select * from regression9_input where lifetime=4 ")
regression9_input_lifetime5 = spark.sql("select * from regression9_input where lifetime=5 ")
regression9_input_lifetime6 = spark.sql("select * from regression9_input where lifetime=6 ")
regression9_input_lifetime7 = spark.sql("select * from regression9_input where lifetime=7 ")
regression9_input_lifetime8 = spark.sql("select * from regression9_input where lifetime=8 ")
regression9_input_lifetime9 = spark.sql("select * from regression9_input where lifetime=9 ")
regression9_input_lifetime10 = spark.sql("select * from regression9_input where lifetime=10 ")
regression9_input_lifetime11 = spark.sql("select * from regression9_input where lifetime=11 ")
regression9_input_lifetime12 = spark.sql("select * from regression9_input where lifetime=12 ")


# In[ ]:


def lifetimeMSE(inputdata):
    Assembler = VectorAssembler(inputCols = ['frequency', 'recency'], 
                                outputCol ='features')
    outputdata = Assembler.transform(inputdata)
    model_df = outputdata.select('features','num_transaction')
    train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=1)
    lin_reg = LinearRegression(labelCol ='num_transaction', featuresCol='features')
    lr_model = lin_reg.fit(train_df)
    test_results = lr_model.evaluate(test_df)
    r_mse = test_results.rootMeanSquaredError
    mse = r_mse**2
    return mse


# In[ ]:


MSE_lifetime0 = lifetimeMSE(regression9_input_lifetime0)
print(MSE_lifetime0)


# In[ ]:


MSE_9 = []
MSE_9.extend([MSE_lifetime0, MSE_lifetime1, MSE_lifetime2, MSE_lifetime3, MSE_lifetime4, MSE_lifetime5, MSE_lifetime6, 
            MSE_lifetime7, MSE_lifetime8, MSE_lifetime9, MSE_lifetime10, MSE_lifetime11, MSE_lifetime12])
MSE_9


# In[ ]:


Lifetime_x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(Lifetime_x, MSE_9, 'ro--', linewidth=2, markersize=8) 


# ## Q10

# In[ ]:


dynamic_profile = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/dynamic_profile.parquet')


# In[ ]:


# Combine the dynamic spending profile table with the frequency and recency table
dynamic_profile.createOrReplaceTempView("dynamic_profile")
regression9_input.createOrReplaceTempView("regression9_input")

full_dynamic = spark.sql('''
                                SELECT user1, lifetime, F.max(frequency) as frequency, \
                                F.max(recency) as recency, F.max(ifnull(people,0)) as people, \
                                F.max(ifnull(food,0)) as food, F.max(ifnull(activity,0)) as activity, \
                                F.max(ifnull(event,0)) as event, F.max(ifnull(travel,0)) as travel, \
                                F.max(ifnull(cash,0)) as cash, F.max(ifnull(utility,0)) as utility, \
                                F.max(ifnull(trasportation,0)) as trasportation, \
                                F.max(ifnull(illegal_sarcasm,0)) as illegal_sarcasm, \
                                F.max(ifnull(not_classified,0)) as not_classified \
                                FROM
                                (SELECT *, null as frequency, null as recency
                                 FROM dynamic_profile
                                 UNION
                                 SELECT user1,lifetime,null,null,null,null,null, \
                                 null,null,null,null,null, frequency, recency 
                                 FROM regression9_input
                                )
                                GROUP BY user1, lifetime
                                ORDER BY user1, lifetime
                                
                                ''')
full_dynamic.createOrReplaceTempView("regression10_input")
#full_dynamic.show()


# In[ ]:


# Generating the regression input table
full_dynamic.createOrReplaceTempView("full_dynamic")
transaction_count.createOrReplaceTempView("transaction_count")
regression10_input = spark.sql('''
                                SELECT * 
                                FROM full_dynamic 
                                JOIN transaction_count
                                USING (user1) 
                                ORDER BY user1, lifetime
                                ''')

regression10_input.createOrReplaceTempView("df_dynamic_input")
regression10_input.head()


# In[ ]:


regression10_input.createOrReplaceTempView("regression10_input")
dynamic_input_time0 = spark.sql("select * from regression10_input where lifetime=0 ")
dynamic_input_time1 = spark.sql("select * from regression10_input where lifetime=1 ")
dynamic_input_time2 = spark.sql("select * from regression10_input where lifetime=2 ")
dynamic_input_time3 = spark.sql("select * from regression10_input where lifetime=3 ")
dynamic_input_time4 = spark.sql("select * from regression10_input where lifetime=4 ")
dynamic_input_time5 = spark.sql("select * from regression10_input where lifetime=5 ")
dynamic_input_time6 = spark.sql("select * from regression10_input where lifetime=6 ")
dynamic_input_time7 = spark.sql("select * from regression10_input where lifetime=7 ")
dynamic_input_time8 = spark.sql("select * from regression10_input where lifetime=8 ")
dynamic_input_time9 = spark.sql("select * from regression10_input where lifetime=9 ")
dynamic_input_time10 = spark.sql("select * from regression10_input where lifetime=10 ")
dynamic_input_time11 = spark.sql("select * from regression10_input where lifetime=11 ")
dynamic_input_time12 = spark.sql("select * from regression10_input where lifetime=12 ")


# In[ ]:


def dynamicMSE(inputdata):
    Assembler = VectorAssembler(inputCols = ['lifetime', 'frequency', 'recency',
                            'people', 'food', 'activity', 'event', 'travel', 'cash',
                            'utility', 'transportation', 'illegal_sarcasm', 'not_classified'], 
                                outputCol ='features')
    output = Assembler.transform(inputdata)
    finalData = output.select('features','num_transaction')
    trainData, testData = finalData.randomSplit([0.7, 0.3], seed=1)
    lrModel = LinearRegression(labelCol ='num_transaction', featuresCol='features')
    lrEstimator = lrModel.fit(trainData)
    testResults = lrEstimator.evaluate(testData)
    rmse = testResults.rootMeanSquaredError
    mse = rmse**2
    return mse


# In[ ]:


dynamic_MSE_time0 = dynamicMSE(dynamic_input_time0)
print("MSE_time0: ", dynamic_MSE_time0)
dynamic_MSE_time1 = dynamicMSE(dynamic_input_time1)
print("MSE_time1: ", dynamic_MSE_time1)
dynamic_MSE_time2 = dynamicMSE(dynamic_input_time2)
print("MSE_time2: ", dynamic_MSE_time2)
dynamic_MSE_time3 = dynamicMSE(dynamic_input_time3)
print("MSE_time3: ", dynamic_MSE_time3)
dynamic_MSE_time4 = dynamicMSE(dynamic_input_time4)
print("MSE_time4: ", dynamic_MSE_time4)
dynamic_MSE_time5 = dynamicMSE(dynamic_input_time5)
print("MSE_time5: ", dynamic_MSE_time5)
dynamic_MSE_time6 = dynamicMSE(dynamic_input_time6)
print("MSE_time6: ", dynamic_MSE_time6)
dynamic_MSE_time7 = dynamicMSE(dynamic_input_time7)
print("MSE_time7: ", dynamic_MSE_time7)
dynamic_MSE_time8 = dynamicMSE(dynamic_input_time8)
print("MSE_time8: ", dynamic_MSE_time8)
dynamic_MSE_time9 = dynamicMSE(dynamic_input_time9)
print("MSE_time9: ", dynamic_MSE_time9)
dynamic_MSE_time10 = dynamicMSE(dynamic_input_time10)
print("MSE_time10: ", dynamic_MSE_time10)
dynamic_MSE_time11 = dynamicMSE(dynamic_input_time11)
print("MSE_time11: ", dynamic_MSE_time11)
dynamic_MSE_time12 = dynamicMSE(dynamic_input_time12)
print("MSE_time12: ", dynamic_MSE_time12)


# In[ ]:


# plot the results
MSE_11 = []
MSE_11.extend([dynamic_MSE_time0, dynamic_MSE_time1, dynamic_MSE_time2, dynamic_MSE_time3, dynamic_MSE_time4, dynamic_MSE_time5, dynamic_MSE_time6, 
            dynamic_MSE_time7, dynamic_MSE_time8, dynamic_MSE_time9, dynamic_MSE_time10, dynamic_MSE_time11, dynamic_MSE_time12])

Lifetime_x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(Lifetime_x, MSE_11, 'go--', linewidth=2, markersize=8) 

plt.xlabel('lifetime')
plt.ylabel('MSE')


# ## Q11

# ### We first run the # friends, # fof, # triangles and page rank. The following code is from the previous sections.

# In[ ]:


dynamic_social_11 = inputdata.select('user1', 'user2', 'datetime').union(inputdata.select('user2', 'user1', 'datetime'))
# use window function to get the date of the first transaction of each user1
from pyspark.sql import Window
window = Window.partitionBy("user1").orderBy("datetime") 
from pyspark.sql.functions import min
dynamic_social_11 = dynamic_social_11.withColumn("start_date", min("datetime").over(window))
# dynamic_social_11.show(5)


# In[ ]:


dynamic_social_11.createOrReplaceTempView("table1")
# calculate the lifetime month of the user1 for each line of transaction
dynamic_social_11 = spark.sql("""SELECT user1, user2, datetime, start_date, 
                           ceil(datediff(datetime, start_date)/30) AS lifetime
                           FROM table1 
                           ORDER BY user1, datetime""")
# dynamic_social_11.show(5)


# In[ ]:


# since we only count the cumulative number of "new" friends met in each lifetime
# get the first 'lifetime' value when each pair of users met
dynamic_social_11.createOrReplaceTempView("table1")
num_friends = spark.sql("""SELECT user1, user2, MIN(lifetime) AS lifetime
                           FROM table1 
                           GROUP BY user1, user2
                           ORDER BY user1, MIN(lifetime)""")
friend_list = num_friends
# num_friends.show(10)


# In[ ]:


# use map reduce to create a dataframe with two columns: userid and lifetime that is consecutive from 0 to 12 months
user_lifetime = dynamic_social_11.select("user1", "lifetime").rdd
user_lifetime = user_lifetime.flatMapValues(lambda value: range(0,13))
# convert user_lifetime to a dataframe and rename the columns
user_lifetime = user_lifetime.toDF(["user1", "lifetime"])
# user_lifetime.show(20)


# In[ ]:


# change the schema of user_lifetime, make two columns as integer type
from pyspark.sql.types import IntegerType
user_lifetime = user_lifetime.select(user_lifetime.user1.cast(IntegerType()), 
                                     user_lifetime.lifetime.cast(IntegerType()))


# In[ ]:


num_friends.createOrReplaceTempView("num_friends")
user_lifetime.createOrReplaceTempView("user_lifetime")
num_friends_one_year = spark.sql("""
                        SELECT user1, lifetime, COUNT(DISTINCT user2) AS num_friends
                        FROM 
                          (SELECT user1, lifetime, user2 FROM num_friends
                          UNION
                          SELECT user1, lifetime, null AS user2 FROM user_lifetime) union_table
                        WHERE lifetime <= 12
                        GROUP BY user1, lifetime
                        ORDER BY user1, lifetime
                        """)
# num_friends_one_year.show(20)


# In[ ]:


# calculate the cumulative number of friends in each lifetime month for each user
num_fnd_window = Window.partitionBy("user1").orderBy("lifetime") 
from pyspark.sql.functions import sum, col
num_friends_one_year = num_friends_one_year.withColumn("cum_num_friends", 
                                                       sum("num_friends").over(num_fnd_window)).sort(col("user1"), 
                                                                                                     col("lifetime"))
# num_friends_one_year.show(15)


# In[ ]:


dynamic_social_11.createOrReplaceTempView("table1")
dynamic_social_11.createOrReplaceTempView("table2")
# self join the dynamic_social table, and select the first lifetime month that user met the friend of friend
dynamic_social_self_join_11 = spark.sql("""SELECT table1.user1, table2.user2 as friend_of_friend,
                                    MIN(table1.lifetime) AS lifetime
                                    FROM table1
                                    LEFT JOIN table2 ON table1.user2 = table2.user1
                                    AND table1.user1 != table2.user2
                                    AND table1.datetime >= table2.datetime
                                    WHERE table1.lifetime <= 12
                                    GROUP BY table1.user1, table2.user2
                                    ORDER BY table1.user1, MIN(table1.lifetime)""")
# dynamic_social_self_join_11.show(15)


# In[ ]:


dynamic_social_self_join_11.createOrReplaceTempView("table1")
friend_list.createOrReplaceTempView("table2")
# only consider the friend of friend whom user1 doesn't know. exclude the 1st degree friend
num_friends_of_friends = spark.sql("""SELECT table1.user1, friend_of_friend, table1.lifetime
                                   FROM table1
                                   LEFT JOIN table2 ON table1.user1 = table2.user1
                                   AND table1.lifetime = table2.lifetime
                                   AND table1.friend_of_friend = table2.user2
                                   WHERE table2.user2 is null
                                   OR friend_of_friend is null
                                   ORDER BY table1.user1, table1.lifetime
                                    """)
friends_of_friends = num_friends_of_friends
# num_friends_of_friends.show(10)


# In[ ]:


num_friends_of_friends.createOrReplaceTempView("num_friends_of_friends")
user_lifetime.createOrReplaceTempView("user_lifetime")
num_friends_of_friends_one_year = spark.sql("""
                        SELECT user1, lifetime, COUNT(DISTINCT friend_of_friend) AS num_fnd_of_fnd
                        FROM 
                          (SELECT user1, lifetime, friend_of_friend FROM num_friends_of_friends
                          UNION
                          SELECT user1, lifetime, null AS friend_of_friend FROM user_lifetime) union_table
                        WHERE lifetime <= 12
                        GROUP BY user1, lifetime
                        ORDER BY user1, lifetime
                        """)
# num_friends_of_friends_one_year.show(20)


# In[ ]:



# use window function to calculate the cumulative number of "new" friends of friends for each lifetime month
num_fnd_fnd_window = Window.partitionBy("user1").orderBy("lifetime") 
from pyspark.sql.functions import sum
num_friends_of_friends_one_year = num_friends_of_friends_one_year.withColumn("cum_num_fnd_of_fnd", 
                                sum("num_fnd_of_fnd").over(num_fnd_fnd_window)).sort(col("user1"), 
                                                                                           col("lifetime"))
# num_friends_of_friends_one_year.show(20)


# In[ ]:


dynamic_social_11.createOrReplaceTempView("t1")
# only consider the date time when two users first transacted
edges = spark.sql("""SELECT user1, user2, MIN(datetime) AS datetime, MIN(lifetime) AS lifetime, MIN(start_date) AS start_date
                      FROM t1 
                      GROUP BY user1, user2
                      ORDER BY user1, MIN(datetime)""")
# edges.show(5)


# In[ ]:


edges.createOrReplaceTempView("t1")
edges.createOrReplaceTempView("t2")
edges.createOrReplaceTempView("t3")
# self join the edges table for three times, and make user2 of the table 3 equal to the user1 of the table1
edges_3join = spark.sql("""SELECT t1.user1 AS t1_user1, t1.user2 AS t1_user2, 
                        t1.start_date AS t1_start_date, t1.datetime AS t1_datetime, 
                        t2.user1 AS t2_user1, t2.user2 AS t2_user2, t2.datetime AS t2_datetime, 
                        t3.user1 AS t3_user1, t3.user2 AS t3_user2, t3.datetime AS t3_datetime
                      FROM t1
                      LEFT JOIN t2 ON t1.user2 = t2.user1
                      AND t1.user1 != t2.user2
                      LEFT JOIN t3 ON t2.user2 = t3.user1
                      AND t2.user1 != t3.user2
                      WHERE t1.user1 == t3.user2
                      ORDER BY t1.user1, t1.lifetime""")
# edges_3join.show(20)
# for table1 user1, each triangle will appear twice in the results table 


# In[ ]:


edges_3join.createOrReplaceTempView("t1")
# calculate the date when the triangle formed by selecting the latest date of the transaction between three users
# then calculate the lifetime for t1_user1 when the triangle formed 
edges_3join_tridate = spark.sql("""
                                SELECT t1_user1, t2_user1, t3_user1,
                                ceil(DATEDIFF(GREATEST(t1_datetime, t2_datetime, t3_datetime), t1_start_date)/30)
                                AS triangle_lifetime
                                FROM t1
                                """)
# edges_3join_tridate.show(5)


# In[ ]:


edges_3join_tridate.createOrReplaceTempView("t1")
# get the triangles of lifetime <= 12, and count the number of triangles group by each user and lifetime
num_triangles = spark.sql("""
                                SELECT t1_user1 AS user1, triangle_lifetime AS lifetime,
                                COUNT(*)/2 AS num_triangles
                                FROM t1
                                WHERE triangle_lifetime <= 12
                                GROUP BY t1_user1, triangle_lifetime
                                ORDER BY t1_user1, triangle_lifetime
                                """)
# num_triangles.show(5)


# In[ ]:


num_triangles.createOrReplaceTempView("t1")
user_lifetime.createOrReplaceTempView("t2")
# get # triangles for consecutive 0-12 months
num_triangles_one_year = spark.sql("""
                                SELECT user1, lifetime, SUM(num_triangles) AS num_triangles
                                FROM(
                                  SELECT user1, lifetime, num_triangles
                                  FROM t1
                                  UNION
                                  SELECT user1, lifetime, 0 AS num_triangles
                                  FROM t2) temp_table
                                GROUP BY user1, lifetime
                                ORDER BY user1, lifetime
                                """)
# num_triangles_one_year.show(5)


# In[ ]:


# calculate the cumulative # triangles in each lifetime month for each user
num_tri_window = Window.partitionBy("user1").orderBy("lifetime") 
from pyspark.sql.functions import sum, col
num_triangles_one_year = num_triangles_one_year.withColumn("cum_num_triangles", 
                                                       sum("num_triangles").over(num_tri_window)).sort(col("user1"), 
                                                                                                     col("lifetime"))
# num_triangles_one_year.show(15)


# ### Now run regression for Q11!

# In[ ]:


transaction_count.createOrReplaceTempView("txn_count")
num_friends_one_year.createOrReplaceTempView("num_friends")
num_friends_of_friends_one_year.createOrReplaceTempView("num_fof")
num_triangles_one_year.createOrReplaceTempView("num_tri")
pr_df.createOrReplaceTempView("pr_df")


# In[ ]:


# join tables to combine user lifetime, # friends, # fof, # of triangles, page rank and total transactions (y)
regression11_input = spark.sql("""SELECT num_friends.user1, num_friends.lifetime, cum_num_friends AS num_friends, 
                  cum_num_fnd_of_fnd AS num_friends_of_friends, 
                  cum_num_triangles AS num_triangles, PageRank AS page_rank, num_transaction
                  FROM num_friends 
                  JOIN num_fof ON num_friends.user1 = num_fof.user1
                  AND num_friends.lifetime = num_fof.lifetime
                  JOIN num_tri ON num_friends.user1 = num_tri.user1
                  AND num_friends.lifetime = num_tri.lifetime
                  JOIN txn_count USING (user1)
                  JOIN pr_df USING (user1)
                  ORDER BY user1, lifetime""")
# regression11_input.show()


# In[ ]:


regression11_input.createOrReplaceTempView("regression11_input")
regression11_input_lifetime0 = spark.sql("select * from regression11_input where lifetime=0 ")
regression11_input_lifetime1 = spark.sql("select * from regression11_input where lifetime=1 ")
regression11_input_lifetime2 = spark.sql("select * from regression11_input where lifetime=2 ")
regression11_input_lifetime3 = spark.sql("select * from regression11_input where lifetime=3 ")
regression11_input_lifetime4 = spark.sql("select * from regression11_input where lifetime=4 ")
regression11_input_lifetime5 = spark.sql("select * from regression11_input where lifetime=5 ")
regression11_input_lifetime6 = spark.sql("select * from regression11_input where lifetime=6 ")
regression11_input_lifetime7 = spark.sql("select * from regression11_input where lifetime=7 ")
regression11_input_lifetime8 = spark.sql("select * from regression11_input where lifetime=8 ")
regression11_input_lifetime9 = spark.sql("select * from regression11_input where lifetime=9 ")
regression11_input_lifetime10 = spark.sql("select * from regression11_input where lifetime=10 ")
regression11_input_lifetime11 = spark.sql("select * from regression11_input where lifetime=11 ")
regression11_input_lifetime12 = spark.sql("select * from regression11_input where lifetime=12 ")


# In[ ]:


# save the monthly regression table as parquet to speed up the future regression
regression11_input_lifetime2.coalesce(1).write.format("parquet").mode("append").save("regression_q11_m2.parquet")


# In[ ]:


get_ipython().system('mv regression_q11_m2.parquet /content/drive/My\\ Drive/ConFiveDance/code')


# In[ ]:


from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# In[ ]:


def lifetimeMSE(inputdata):
    Assembler = VectorAssembler(inputCols = ['lifetime', 'num_friends', 'num_friends_of_friends',
                                             'num_triangles', 'page_rank'], 
                                outputCol ='features')
    outputdata = Assembler.transform(inputdata)
    model_df = outputdata.select('features','num_transaction')
    train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=1)
    lin_reg = LinearRegression(labelCol ='num_transaction', featuresCol='features')
    lr_model = lin_reg.fit(train_df)
    test_results = lr_model.evaluate(test_df)
    r_mse = test_results.rootMeanSquaredError
    mse = r_mse**2
    return mse


# In[ ]:


MSE_lifetime0 = lifetimeMSE(regression11_input_lifetime0)
print("MSE of lifetime 0 is:", MSE_lifetime0)

MSE_lifetime1 = lifetimeMSE(regression11_input_lifetime1)
print("MSE of lifetime 1 is:", MSE_lifetime1)

MSE_lifetime2 = lifetimeMSE(regression11_input_lifetime2)
print("MSE of lifetime 2 is:",MSE_lifetime2)

MSE_lifetime3 = lifetimeMSE(regression11_input_lifetime3)
print("MSE of lifetime 3 is:", MSE_lifetime3)

MSE_lifetime4 = lifetimeMSE(regression11_input_lifetime4)
print("MSE of lifetime 4 is:", MSE_lifetime4)

MSE_lifetime5 = lifetimeMSE(regression11_input_lifetime5)
print("MSE of lifetime 5 is:", MSE_lifetime5)

MSE_lifetime6 = lifetimeMSE(regression11_input_lifetime6)
print("MSE of lifetime 6 is:", MSE_lifetime6)

MSE_lifetime7 = lifetimeMSE(regression11_input_lifetime7)
print("MSE of lifetime 7 is:", MSE_lifetime7)

MSE_lifetime8 = lifetimeMSE(regression11_input_lifetime8)
print("MSE of lifetime 8 is:", MSE_lifetime8)

MSE_lifetime9 = lifetimeMSE(regression11_input_lifetime9)
print("MSE of lifetime 9 is:", MSE_lifetime9)

MSE_lifetime10 = lifetimeMSE(regression11_input_lifetime10)
print("MSE of lifetime 10 is:", MSE_lifetime10)

MSE_lifetime11 = lifetimeMSE(regression11_input_lifetime11)
print("MSE of lifetime 11 is:", MSE_lifetime11)

MSE_lifetime12 = lifetimeMSE(regression11_input_lifetime12)
print("MSE of lifetime 12 is:", MSE_lifetime12)


# In[ ]:


# standardize PageRank so that the coefficients and standard errors of the variables would be on the same scale
regression11_input_lifetime12.createOrReplaceTempView('q11')
q11_std_input = spark.sql(""" select lifetime, num_friends, num_friends_of_friends,
num_triangles, (page_rank - avg(page_rank) over())/(std(page_rank) over()) AS std_page_rank, num_transaction
FROM q11
""")


# In[ ]:


# Run the regression lifetime 12, get the MSE and and see the important features
Assembler = VectorAssembler(inputCols = [ 'num_friends', 'num_friends_of_friends',
                                          'num_triangles', 'page_rank'], 
                            outputCol ='features')
outputdata = Assembler.transform(q11_std_input)
model_df = outputdata.select('features','num_transaction')
train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=1)
lin_reg = LinearRegression(labelCol ='num_transaction', featuresCol='features')
lr_model = lin_reg.fit(train_df)
test_results = lr_model.evaluate(test_df)
r_mse = test_results.rootMeanSquaredError
MSE_lifetime12 = r_mse**2
trainingSummary = lrModel.summary


# In[ ]:


print("Coefficients: %s" % str(lr_model.coefficients))
print("P Values: " + str(trainingSummary.pValues))
print("Coefficient Standard Errors: " + str(trainingSummary.coefficientStandardErrors))


# In[ ]:


MSE_11 = []
MSE_11.extend([MSE_lifetime0, MSE_lifetime1, MSE_lifetime2, MSE_lifetime3, MSE_lifetime4, MSE_lifetime5, MSE_lifetime6, 
            MSE_lifetime7, MSE_lifetime8, MSE_lifetime9, MSE_lifetime10, MSE_lifetime11, MSE_lifetime12])
MSE_11


# In[ ]:


Lifetime_x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(Lifetime_x, MSE_11, 'ro--', linewidth=2, markersize=8) 


# ## Q12

# In[ ]:


# read the output of Q4 that we saved as parquet
cat_parquet = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/dynamic_profile.parquet')


# In[ ]:


# calculate the running average of spending profile to make the spending cumulative
cat_parquet.createOrReplaceTempView('spend_prof')
running_avg_sp = spark.sql(""" SELECT user1, lifetime, 
AVG(activity) OVER (PARTITION BY user1 ORDER BY lifetime) AS Activity,
AVG(Cash) OVER (PARTITION BY user1 ORDER BY lifetime) AS Cash,
AVG(Event) OVER (PARTITION BY user1 ORDER BY lifetime) AS Event,
AVG(Food) OVER (PARTITION BY user1 ORDER BY lifetime) AS Food,
AVG(Illegal_Sarcasm) OVER (PARTITION BY user1 ORDER BY lifetime) AS Illegal_Sarcasm,
AVG(Not_classified) OVER (PARTITION BY user1 ORDER BY lifetime) AS Not_classified,
AVG(People) OVER (PARTITION BY user1 ORDER BY lifetime) AS People,
AVG(Transportation) OVER (PARTITION BY user1 ORDER BY lifetime) AS Transportation,
AVG(Travel) OVER (PARTITION BY user1 ORDER BY lifetime) AS Travel,
AVG(Utility) OVER (PARTITION BY user1 ORDER BY lifetime) AS Utility
FROM spend_prof ORDER BY user1, lifetime
""")
running_avg_sp.show(20)


# In[ ]:


dynamic_social_12 = inputdata.select('user1', 'user2', 'datetime').union(inputdata.select('user2', 'user1', 'datetime'))


# In[ ]:


# use window function to get the date of the first transaction of each user1
window1 = Window.partitionBy("user1").orderBy("datetime") 
window2 = Window.partitionBy("user2").orderBy("datetime") 
dynamic_social_12 = dynamic_social.withColumn("user1_start_date", 
                                              min("datetime").over(window1)).withColumn("user2_start_date", 
                                                                                       min("datetime").over(window2))
# dynamic_social_12.show(5)


# In[ ]:


dynamic_social_12.createOrReplaceTempView("table1")
# calculate the lifetime month of the user1 for each line of transaction
ds_12 = spark.sql("""SELECT user1, user2, datetime, user1_start_date, 
                           ceil(datediff(datetime, user1_start_date)/30) AS user1_lifetime,
                           user2_start_date,
                           ceil(datediff(datetime, user2_start_date)/30) AS user2_lifetime
                           FROM table1
                           WHERE ceil(datediff(datetime, user1_start_date)/30) <=12 
                           """)
ds_12.show(5)


# In[ ]:


ds_12.createOrReplaceTempView("t1")
running_avg_sp.createOrReplaceTempView("t2")

reg_12 = spark.sql("""SELECT t1.user1, t1.user2, t1.user1_lifetime AS lifetime,
                    Activity, Cash, Event, Food, Illegal_Sarcasm, Not_classified,
                    People, Transportation, Travel, Utility
                    FROM t1
                    JOIN t2 ON t1.user2 = t2.user1 
                    AND t1.user2_lifetime = t2.lifetime
                    """)


# In[ ]:


# read the regression input of the Q11 lifetime 0 we saved as parquet 
reg_11_m0 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/regression_q11_m0.parquet')
reg_11_m0.show(5)


# In[ ]:


reg_12.createOrReplaceTempView("t1")
reg_11_m0.createOrReplaceTempView("t2")
reg_q12_m0 = spark.sql("""
                      SELECT *
                      FROM t1
                      JOIN t2 USING (user1, lifetime) """)
reg_q12_m0.show(5)


# In[ ]:


reg_q12_m0.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m0.parquet")
get_ipython().system('mv reg_q12_m0.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m1.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m1.parquet")
get_ipython().system('mv reg_q12_m1.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m2.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m2.parquet")
get_ipython().system('mv reg_q12_m2.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m3.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m3.parquet")
get_ipython().system('mv reg_q12_m3.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m4.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m4.parquet")
get_ipython().system('mv reg_q12_m4.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m5.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m5.parquet")
get_ipython().system('mv reg_q12_m5.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m6.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m6.parquet")
get_ipython().system('mv reg_q12_m6.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m7.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m7.parquet")
get_ipython().system('mv reg_q12_m7.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m8.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m8.parquet")
get_ipython().system('mv reg_q12_m8.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m9.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m9.parquet")
get_ipython().system('mv reg_q12_m9.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m10.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m10.parquet")
get_ipython().system('mv reg_q12_m10.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m11.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m11.parquet")
get_ipython().system('mv reg_q12_m11.parquet /content/drive/My\\ Drive/ConFiveDance/code')

reg_q12_m12.coalesce(1).write.format("parquet").mode("append").save("reg_q12_m12.parquet")
get_ipython().system('mv reg_q12_m12.parquet /content/drive/My\\ Drive/ConFiveDance/code')


# In[ ]:


regression_input_q12_m0 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m0.parquet')
regression_input_q12_m1 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m1.parquet')
regression_input_q12_m2 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m2.parquet')
regression_input_q12_m3 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m3.parquet')
regression_input_q12_m4 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m4.parquet')
regression_input_q12_m5 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m5.parquet')
regression_input_q12_m6 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m6.parquet')
regression_input_q12_m7 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m7.parquet')
regression_input_q12_m8 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m8.parquet')
regression_input_q12_m9 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m9.parquet')
regression_input_q12_m10 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m10.parquet')
regression_input_q12_m11 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m11.parquet')
regression_input_q12_m12 = spark.read.parquet('/content/drive/My Drive/ConFiveDance/code/reg_q12_m12.parquet')


# In[ ]:


def lifetimeMSE(inputdata):
    Assembler = VectorAssembler(inputCols = ['Activity', 'Cash', 'Event', 'Food', 'Illegal_Sarcasm',
                                              'Not_classified', 'People', 'Transportation', 'Travel', 'Utility',
                                              'num_friends', 'num_friends_of_friends',
                                             'num_triangles', 'page_rank'], 
                                outputCol ='features')
    outputdata = Assembler.transform(inputdata)
    model_df = outputdata.select('features','num_transaction')
    train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=1)
    lin_reg = LinearRegression(labelCol ='num_transaction', featuresCol='features')
    lr_model = lin_reg.fit(train_df)
    test_results = lr_model.evaluate(test_df)
    r_mse = test_results.rootMeanSquaredError
    mse = r_mse**2
    return mse


# In[ ]:


MSE_lifetime0 = lifetimeMSE(regression_input_q12_m0)
print("MSE of lifetime 0 is:", MSE_lifetime0)

MSE_lifetime1 = lifetimeMSE(regression_input_q12_m1)
print("MSE of lifetime 1 is:", MSE_lifetime1)

MSE_lifetime2 = lifetimeMSE(regression_input_q12_m2)
print("MSE of lifetime 2 is:",MSE_lifetime2)

MSE_lifetime3 = lifetimeMSE(regression_input_q12_m3)
print("MSE of lifetime 3 is:", MSE_lifetime3)

MSE_lifetime4 = lifetimeMSE(regression_input_q12_m4)
print("MSE of lifetime 4 is:", MSE_lifetime4)

MSE_lifetime5 = lifetimeMSE(regression_input_q12_m5)
print("MSE of lifetime 5 is:", MSE_lifetime5)

MSE_lifetime6 = lifetimeMSE(regression_input_q12_m6)
print("MSE of lifetime 6 is:", MSE_lifetime6)

MSE_lifetime7 = lifetimeMSE(regression_input_q12_m7)
print("MSE of lifetime 7 is:", MSE_lifetime7)

MSE_lifetime8 = lifetimeMSE(regression_input_q12_m8)
print("MSE of lifetime 8 is:", MSE_lifetime8)

MSE_lifetime9 = lifetimeMSE(regression_input_q12_m9)
print("MSE of lifetime 9 is:", MSE_lifetime9)

MSE_lifetime10 = lifetimeMSE(regression_input_q12_m10)
print("MSE of lifetime 10 is:", MSE_lifetime10)

MSE_lifetime11 = lifetimeMSE(regression_input_q12_m11)
print("MSE of lifetime 11 is:", MSE_lifetime11)

MSE_lifetime12 = lifetimeMSE(regression_input_q12_m12)
print("MSE of lifetime 12 is:", MSE_lifetime12)


# In[ ]:


MSE_12 = []
MSE_12.extend([MSE_lifetime0, MSE_lifetime1, MSE_lifetime2, MSE_lifetime3, MSE_lifetime4, MSE_lifetime5, MSE_lifetime6, 
            MSE_lifetime7, MSE_lifetime8, MSE_lifetime9, MSE_lifetime10, MSE_lifetime11, MSE_lifetime12])
MSE_12


# In[ ]:


Lifetime_x = [0,1,2,3,4,5,6,7,8,9,10,11,12]
plt.plot(Lifetime_x, MSE_12, 'mo--', linewidth=2, markersize=8) 

plt.xlabel('lifetime')
plt.ylabel('MSE')

