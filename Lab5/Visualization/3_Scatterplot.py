## Bước #1: Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""## Bước #2: Đọc dữ liệu"""
df = pd.read_csv("https://raw.githubusercontent.com/ltdaovn/dataset/master/turtle.csv", delimiter='\t')
df.head()
#df.describe()

"""## Bước #3a: vẽ scatter plot thể hiện mối tương quan giữa 2
thuộc tính width và height với pandas!"""
df.plot.scatter(x = 'width', y = 'height')

"""## Bước #3b: vẽ scatter plot thể hiện mối tương quan giữa 2
thuộc tính width và height với matplotlib!"""
x = df.width
y = df.height
plt.scatter(x,y)

"""## Bước #4a: Tính ma trận tương quan giữa các biến"""
df.corr()

"""## Bước #4b: Vẽ ma trận tương quan giữa các biến"""
plt.figure(figsize=(6, 5))

sns.heatmap(df.corr(), annot=True)
# sns.heatmap(df.corr())
plt.show()