"Instagram Reach Forecasting using Python"

step1:Importing the necessary Python libraries and the dataset:

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
data = pd.read_csv("Instaa.txt", encoding = 'latin-1')
print(data.head())

step2:I’ll convert the Date column into datetime datatype to move forward:

data['Date'] = pd.to_datetime(data['Date'])
print(data.head())

"Analyzing Reach"
#Let’s analyze the trend of Instagram reach over time using a line chart:

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], 
                         y=data['Instagram reach'], 
                         mode='lines', name='Instagram reach'))
fig.update_layout(title='Instagram Reach Trend', xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()


"Now analyze Instagram reach for each day using a BAR CHART":

fig = go.Figure()
fig.add_trace(go.Bar(x=data['Date'], 
                     y=data['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach by Day', 
                  xaxis_title='Date', 
                  yaxis_title='Instagram Reach')
fig.show()


"Now analyze the distribution of Instagram reach using a BOX PLOT":

fig = go.Figure()
fig.add_trace(go.Box(y=data['Instagram reach'], 
                     name='Instagram reach'))
fig.update_layout(title='Instagram Reach Box Plot', 
                  yaxis_title='Instagram Reach')
fig.show()


"analyze reach based on the Days of the Week":

data['Day'] = data['Date'].dt.day_name()
print(data.head())


#Calculate the mean, median, and standard deviation of the Instagram reach column for each day:

import numpy as np
import numpy as np
day_stats = data.groupby('Day')['Instagram reach'].agg(['mean', 'median', 'std']).reset_index()
print(day_stats)

# A bar chart to visualize the reach for each day of the week:

fig = go.Figure()
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['mean'], 
                     name='Mean'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['median'], 
                     name='Median'))
fig.add_trace(go.Bar(x=day_stats['Day'], 
                     y=day_stats['std'], 
                     name='Standard Deviation'))
fig.update_layout(title='Instagram Reach by Day of the Week', 
                  xaxis_title='Day', 
                  yaxis_title='Instagram Reach')
fig.show()


# Instagram Reach Forecasting using Time Series Forecasting:


from plotly.tools import mpl_to_plotly
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = data[["Date", "Instagram reach"]]

result = seasonal_decompose(data['Instagram reach'], 
                            model='multiplicative', 
                            period=100)

fig = plt.figure()
fig = result.plot()

fig = mpl_to_plotly(fig)
fig.show()


#Now here’s how to visualize an autocorrelation plot to find the value of p:

pd.plotting.autocorrelation_plot(data["Instagram reach"])


# And now here’s how to visualize a partial autocorrelation plot to find the value of q:

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(data["Instagram reach"], lags=100, method='ywm')

