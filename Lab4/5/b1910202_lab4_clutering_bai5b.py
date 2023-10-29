# Nạp các gói thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
# 1. Chuẩn bị dữ liệu
# Đọc dữ liệu từ tập tin csv
df = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/Eurojobs.csv')
# Lấy dữ liệu thu nhập hằng năm (Annual Income) và điểm thành viên (SpendingScore) để phân lớp
X = df.iloc[:, [1,2,3,4,5,6,7,8,9]].values
y = df.iloc[:,0].values
#print(X)
# 2. Tiến hành gom nhóm
# Khi tiến hành các giải thuật để gom nhóm,
# câu hỏi đặt ra là với dataset đã có,
# chúng ta sẽ phân thành bao nhiêu cụm là hợp lý (tối ưu)?
# Trong ví dụ này chúng ta sẽ sử dụng Dendrogram để xác định số cụm.
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'),  labels=y)
plt.title('Dendrogram')
plt.xlabel('Quốc gia')
plt.ylabel('Khoảng cách Euclidean')
plt.show()