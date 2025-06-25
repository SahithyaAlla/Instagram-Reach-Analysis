#"Instagram Reach Analysis"
"Now let’s start the task of analyzing the reach of my Instagram account by importing the necessary Python libraries and the dataset:"

get_ipython().system('pip install plotly')
get_ipython().system('pip install wordcloud')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
print(data.head())

#"step2:Before starting everything, let’s have a look at whether this dataset contains any null values or not:"

data.isnull().sum()


#"step3:So it has a null value in every column. Let’s drop all these null values and move further:"


data = data.dropna()


#"step4:Let’s have a look at the insights of the columns to understand the data type of all the columns:"

data.info()


#Now let’s start with analyzing the reach of my Instagram posts. I will first have a look at the distribution of impressions I have received from home:"

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'])
plt.show()


#let’s have a look at the distribution of the impressions I received from hashtags:

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'])
plt.show()


#let’s have a look at the distribution of impressions I have received from the explore section of Instagram:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'])
plt.show()


#let’s have a look at the percentage of impressions I get from various sources on Instagram:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


#Analyzing Content
#Now let’s analyze the content of my Instagram posts.

text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#through hashtag

text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#Analyzing Relationships


get_ipython().system('pip install statsmodels')
import plotly.express as px

#"Relationship Between Likes and Impressions":

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


#"Relationship Between Comments and Total Impressions"


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


#"Relationship Between Shares and Total Impressions"


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")
figure.show()


#"Relationship Between Post Saves and Total Impressions"


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()


#Instagram Reach Prediction Model:


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


#Now here’s is how we can train a machine learning model to predict the reach of an Instagram post using Python:

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


#Now let’s predict the reach of an Instagram post by giving inputs to the machine learning model:


# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)





