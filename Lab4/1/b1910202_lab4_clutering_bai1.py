from sklearn.cluster import KMeans
import numpy as np
# Khởi động 2 tâm
my_centroids = np.array([[1, 1], [2, 1]])
# Các điểm dữ liệu cần gom nhóm
data = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])
# Thực hiện gom nhóm với KMeans
kmeans = KMeans(n_clusters=2, random_state=0, init=my_centroids, n_init=10).fit(data)
# Hiển thị kết quả gom nhóm
kmeans.labels_
# Hiển thị tâm của các nhóm sau khi đã được gom
kmeans.cluster_centers_
# Dự đoán nhóm cho các phần tử mới
kmeans.predict([[1, 2], [4, 4]])