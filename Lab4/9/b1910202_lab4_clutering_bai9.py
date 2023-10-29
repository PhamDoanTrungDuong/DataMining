import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/ltdaovn/dataset/master/bank-data.csv"
data = pd.read_csv(url)

# Hiển thị một số mẫu dữ liệu ban đầu
print(data.head())

# Chọn các thuộc tính bạn muốn sử dụng cho phân tích cụm
# Ví dụ: chọn các thuộc tính 'age' và 'income'
X = data[['age', 'income']]

# Chuẩn hóa dữ liệu để đảm bảo cùng tỷ lệ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sử dụng K-Means để phân tích cụm và vẽ biểu đồ Elbow
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Biểu đồ Elbow')
plt.xlabel('Số lượng cụm')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Dựa vào biểu đồ Elbow, bạn có thể chọn số cụm tối ưu, ví dụ: 3.

# Sử dụng số cụm tối ưu (ví dụ: 3) để thực hiện phân tích cụm bằng K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# Thêm nhãn cụm vào DataFrame ban đầu
data['Cluster'] = kmeans.labels_

# Vẽ biểu đồ Scatter
plt.figure(figsize=(10, 6))
plt.scatter(data['age'], data['income'], c=data['Cluster'], cmap='viridis')
plt.title('Biểu đồ phân tán với K-Means Clustering')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()


