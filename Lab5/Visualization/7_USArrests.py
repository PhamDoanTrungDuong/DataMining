import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/ltdaovn/dataset/master/USArrests.csv"
data = pd.read_csv(url, index_col=0)

# Tạo ma trận tương quan
correlation_matrix = data.corr()

# Vẽ biểu đồ heatmap để hiển thị tương quan
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Mối tương quan giữa các thuộc tính trong USArrests")
plt.show()
