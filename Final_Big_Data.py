#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pandas as pd
import numpy as np


# #Question
# Is there a connection between happiness of citizens in a country and several social and financial 
# metrics?

# In[15]:


#using panda's funciton read_csv to read the data from the CountriesHappiness csv file
df = pd.read_csv("CountriesHapiness.csv")
# Outputting Data frame
df


# In[ ]:





# In[16]:


#Summary of the data(Count,Mean,std,min,max...)
df.describe()


# # Dropping unrelated columns
# For this we will basically find the columns that are not related to Happiness Score. So to do this we will create a correlation matrix to find out that.
# 

# In[17]:


import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
  
  
# plotting correlation heatmap
#dataframe.corr() is used to find the pairwise correlation of all columns
#Color type cmap="YlGnBu"
#annot If True, write the data value in each cell
#linewidths Width of the lines that will divide each cell
dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True,linewidths=.8)
# print(df.corr())
# displaying heatmap
mp.show()
#Correlation ranges from -1 to +1. Values closer to zero means there is no linear trend between the two variables. The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other and the closer to 1 the stronger this relationship is. A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases


# Our main variable is the Happiness Score
# So, according to the analysis the needed columns are the ones that have a good effective correlation with base column.
# 
# not good are:
# 
# 1. Explained by: Perceptions of corruption
# 
# 2. RANK
# 
# 3. Country
# 
# 4. Explained by: Generosity
# 
# 5. Dystopia (1.83) + residual
# 
# these variables are removed from the dataset because these are not effectively correlated to each other(with the main variable).
# 
# 
# 

# In[18]:


# we deleted these columns because they are least correlated with the base column (Happiness Score) 
df=df.drop(columns=['Explained by: Perceptions of corruption','RANK','Country'])
df=df.drop(columns=['Explained by: Generosity','Dystopia (1.83) + residual'])


# # Data Pre-processing
# 

# ### Finding NULL VALUES
# 

# In[19]:


df.isna().sum()


# Well, there are no any null values, lets move to outliers

# ## Finding Outliers
# 

# In[20]:


print(df.columns)
df['Happiness score'].hist()


# Data is well distributed there are not outliers in the dataset Happiness Score Column

# In[21]:


import plotly.express as px
#For each column we make an histogram to see if there is actually a gaps between the values that are outliers
for each in df.columns:
  fig = px.histogram(df, x=each)
  fig.show()


# 
# Some outliers between the values in the GDP per capatia, and Explained by: Health Expectancy We will replace them with the average value. As we can see that there is the GAP between some occurences, so we have to normalize the data by replacing the outliers with the average.
# 
# 

# # Finding P-values

# In[22]:


from scipy import stats
p_df=pd.DataFrame(columns=['correlation','p-values'])
for col in df.columns:
  if col != 'Happiness score':
    r,s=stats.pearsonr(df['Happiness score'],df[col])
    p_df.loc[col]=[round(r,3),round(s,100)]

p_df  
#Data frame with cols names and correlation betnwen col and base col with 3 digitals and pvalue with 100 digits


# This is the p value and the correlation value, as p-value tell the propbability accurance of the values, the factors that most effects are the whisker high, whisker low, the Explained by GDP, Social support ...
# 

# In[69]:


#Confidence intervals
import pandas as pd
import numpy as np
import math
for each in df.columns:
  if each != 'Happiness score':
    stats = df.groupby(['Happiness score'])[each].agg(['mean', 'count'])
    #ci for confidence interval,hi mean the right line and lo mean the low line 
    ci95_high = []
    ci95_low = []
    #Math equation for confidence intervals
    #std of one value equal 0
    for i in stats.index:
      m, c= stats.loc[i]
      ci95_high.append(round(m - scipy.stats.norm.ppf(0.025),3))
      ci95_low.append(round(m + scipy.stats.norm.ppf(0.025),3))
        #norm return minus value
stats['ci95_high'] = ci95_high
stats['ci95_low'] = ci95_low
stats


# Above are the mean values on the standard intervals with the confidence of 95%, we applied the formula, to find the interval of 95% confidence and the values of happiness scores are given above in the column of hi and low value which can occur in that intervals

# # Imputing Outliers by mean of the dataframe

# In[71]:


def impute_outliers_IQR(df):
  q1=df.quantile(0.25)
  q3=df.quantile(0.75)
  rg=q3-q1
  upper = df[~(df>(q3+1.5*rg))].max()
  lower = df[~(df<(q1-1.5*rg))].min()
  #Now, if the value is greater than Q3, or less the Q1, it will be replaced with the mean or average.
  df = np.where(df > upper,df.mean(),np.where(df < lower,df.mean(),df)) #Replace few lines with one
  return df


# #We basically replace the outliers with the mean. This piece of code above basically calculaltes the range and which passes below and up from that range the value is replaced by the mean of the happiness score and other columns 

# In[27]:


columns=['Happiness score', 'Whisker-high', 'Whisker-low',
        'Explained by: GDP per capita',
       'Explained by: Social support', 'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       ]
for each in columns:
  df[each]=impute_outliers_IQR(df[each])

df
#data frame after the outliers replacement
  


# ### Verifying data

# In[72]:


import plotly.express as px
for each in df.columns:
  fig = px.histogram(df, x=each)
  fig.show()


# Now Data is all good to go By comparing the above analysis and graps, after normalizing or imputing the outliers. We can see that there are no gaps present in the series of the data and no outliers like before.

# # Splitting the data

# In[73]:


X=df[['Whisker-high', 'Whisker-low',
       'Explained by: GDP per capita',
       'Explained by: Social support', 'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices']]
y=df['Happiness score']


# In[74]:


#method for the linear regression
def linear_regression(x, y):     
    x_mean = x.mean()
    y_mean = y.mean()
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    B0 = y_mean - (B1*x_mean)
    #it gives the regression line(the equation for fast plotting)
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 3))
    return (B0, B1, reg_line)


# In[75]:


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
for each in X.columns:
  B0, B1, reg_line = linear_regression(df[each], y)
#Giving good understandable graphs and plots
  print('Regression Line: ', reg_line)
  plt.figure(figsize=(12,5))
  plt.scatter(df[each], y, s=300, linewidths=1, edgecolor='black')

  plt.title('How '+each+' Affects Happiness')
  plt.xlabel('Happiness Score', fontsize=15)
  plt.ylabel(''+each, fontsize=15)
  plt.plot(df[each], B0 + B1*df[each], c = 'r', linewidth=5, alpha=.5, solid_capstyle='round')
  plt.scatter(x=df[each].mean(), y=y.mean(), marker='*', s=10**2.5, c='r') # average point
  plt.show()


# # Concluion of linear Regression
# This is the slope of the regression that is calculated on each of the happiness score as x-axis and the columns on by one as the y-axis. Well. we get the straight regression line which means that, these variabels are the ones which are the main factors to bring the happiness.
# 
# 

# # Doing Clustering

# In[ ]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
model = KElbowVisualizer(KMeans(), k=10)
model.fit(X)
model.show()


# In[ ]:


#The method above is showing how much number of clusters can this dataset support. So, there are the
#Optimally 4 clusters inside the dataframe


# # Conclusion-Final
# 
# In the above analysis, we find out that these factors has some strong effects on the Happiness in the country. These are the 
# - Whisker-high
# 
# - Whisker-low	
# 
# - Explained by: GDP per capita	
# 
# - Explained by: Social support	
# 
# - Explained by: Healthy life expectancy	
# 
# - Explained by: Freedom to make life choices
# 
# As they are strong correlated to each other. In the regression line, they also showed up with the great impact. Well, if we see in real-world, the people will be more happy if they have money, if they have social support like some good and caring people, they have good facilities and the goverment support the decision of thers. So, they will be more happier than any country in the world.
# 
# The countries which lack this are in the end postitions of these 146 rows of data. 
# 
