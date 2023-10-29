import pandas as pd
import pyodbc
 
df = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/dataCustomerRFM.csv')
df.head()

from datetime import datetime
# 1. Calculate Recency
# 1.1. Find the most recent orderDate.

dfRecentOrder = pd.pivot_table(data = df, 
               index = ['CustomerID'],
               values = ['OrderDate'],
               aggfunc = {'OrderDate':max}
              )

dfRecentOrder.columns = ['RecentOrderDate']
df = pd.merge(df, dfRecentOrder.reset_index(), on = ['CustomerID'])
df['RecentOrderDate'] = df['RecentOrderDate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df['Recency'] = df['RecentOrderDate'].apply(lambda x: (datetime.now() - x).days)

# Đổi dấu recency
df['Recency'] = - df['Recency']

# 2. Calculate Frequency
dfFrequency = df.groupby('CustomerID').OrderID.nunique().to_frame()
dfFrequency.columns = ['Frequency']
df = pd.merge(df, dfFrequency.reset_index(), on = 'CustomerID')

# 3. Calculate Monetary
dfMonetary = df.groupby('CustomerID').Amount.sum().to_frame()
dfMonetary.columns = ['Monetary']
df = pd.merge(df, dfMonetary.reset_index(), on = 'CustomerID')

orderFrequencies = df['Frequency'].rank(method='first')
df['rFrequency'] = pd.qcut(orderFrequencies, 10, labels = False)
df[['rRecency', 'rMonetary']] = df[['Recency', 'Monetary']].apply(lambda x: pd.qcut(x, 10, labels = False))
df['rank'] = (df['rFrequency'] + df['rRecency'] + df['rMonetary'])/3
df['FinalRank'] = df['rank'].apply(int)

import matplotlib.pyplot as plt

df['rank'].plot.hist(bins = 10)
plt.show()

# help(pd.qcut)
# df['rank'].min()

df['Segment'] = 'Low'
df.loc[(df['rank'] < 7) & (df['rank'] >= 4), 'Segment'] = 'Normal'
df.loc[df['rank'] >= 7, 'Segment'] = 'VIP'

df.groupby('Segment').CustomerID.count().plot.pie(autopct = '%.2f%%', figsize = (8, 8))
plt.title('Tỷ lệ số lượng khách hàng theo segment KH')

df.groupby('Segment').Amount.sum().plot.pie(autopct = '%.2f%%', figsize = (8, 8))
plt.title('Tỷ lệ doanh số theo segment KH')

df.groupby('Segment').Amount.mean().plot.bar(figsize = (6, 8))
plt.title('Doanh số theo segment KH')

df.groupby('Segment').Frequency.mean().plot.bar(figsize = (6, 8))
plt.title('Tần suất mua hàng theo segment KH')

df.groupby('Segment').Recency.mean().plot.bar(figsize = (6, 8))
plt.title('Số ngày mua hàng gần nhất theo segment KH')