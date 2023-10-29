import pandas as pd

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/ltdaovn/dataset/master/flowers.csv"
data = pd.read_csv(url)

# Hiển thị một số mẫu dữ liệu ban đầu
print(data.head())

features = data[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dùng phương pháp Elbow để xác định số cụm tối ưu
distortions = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    distortions.append(kmeans.inertia_)

# Vẽ đồ thị Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, distortions, marker='o')
plt.title('Phương pháp Elbow để chọn số cụm tối ưu')
plt.xlabel('Số lượng cụm')
plt.ylabel('Distortion (SSE)')
plt.show()

# Sử dụng KMeans để gom nhóm dữ liệu
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(features)

# Thêm nhãn cụm vào DataFrame ban đầu
data['Cluster'] = kmeans.labels_

# Tạo biểu đồ phân tán với biểu đồ scatter
plt.figure(figsize=(10, 6))

# Tạo biểu đồ scatter cho từng cụm
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['SepalLength'], cluster_data['SepalWidth'], label=f'Cluster {cluster}')

# Vẽ tâm của các cụm
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('Biểu đồ phân tán với KMeans Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()
