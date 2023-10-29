import seaborn as sns
import matplotlib.pyplot as plt

# Load dữ liệu 'tips'
tips = sns.load_dataset('tips')

# Vẽ biểu đồ Scatter Plot
plt.figure(figsize=(10, 6))  # Đặt kích thước biểu đồ
sns.scatterplot(x='size', y='total_bill', data=tips, alpha=0.5)
plt.title('Mối quan hệ giữa kích thước và tổng số hóa đơn')
plt.xlabel('Kích thước (Size)')
plt.ylabel('Tổng số hóa đơn (Total Bill)')
plt.show()
