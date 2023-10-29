import seaborn as sns
import matplotlib.pyplot as plt

# Load dữ liệu 'diamonds'
diamonds = sns.load_dataset('diamonds')

# Vẽ biểu đồ Scatter Plot
plt.figure(figsize=(10, 6))  # Đặt kích thước biểu đồ
sns.scatterplot(x='carat', y='price', data=diamonds, alpha=0.5)
plt.title('Mối quan hệ giữa trọng lượng và giá kim cương')
plt.xlabel('Trọng lượng (Carat)')
plt.ylabel('Giá (Price)')
plt.show()
