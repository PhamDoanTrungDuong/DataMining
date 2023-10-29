import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/USArrests.csv')
df = df.drop(columns=df.columns[0:1],  index=0)
from sklearn.preprocessing import scale
df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
print(df)

X = df.iloc[:, [0,1,2,3]].values

from sklearn.cluster import KMeans
# clusters = []
# for i in range(1, 10):
#     km = KMeans(n_clusters=i).fit(X)
#     clusters.append(km.inertia_)
import matplotlib.pyplot as plt
# import seaborn as sns
# fig, ax = plt.subplots(figsize=(12, 8))
# sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)
# ax.set_title('Đồ thị Elbow')
# ax.set_xlabel('Số lượng nhóm')
# ax.set_ylabel('Giá trị Inertia')
# plt.show()
# plt.cla()
# Qua đồ thị trên, chúng ta thấy số lượng cluster thích hợp là từ 2 đến 4
# clusters
# Phân tích dữ liệu được gom thành 4 nhóm
km4 = KMeans(n_clusters=4)
y_means = km4.fit_predict(X)
#print(y_means)
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100, c = 'pink',label = 'Nhóm 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100, c = 'yellow',label = 'Nhóm 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100, c = 'red',label = 'Nhóm 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100, c = 'green',label = 'Nhóm 4')

# plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.legend()
plt.grid()
plt.show()
