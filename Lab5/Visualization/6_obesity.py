import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/ltdaovn/dataset/master/obesity.csv"
data = pd.read_csv(url)
# =====================a====================
# Vẽ box plot tỷ trọng mỡ theo giới tính
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='pcfat', data=data)
plt.title('Box Plot tỷ trọng mỡ theo giới tính')
plt.xlabel('Giới tính')
plt.ylabel('Tỷ trọng mỡ (pcfat)')

# =====================b====================
# Tạo các nhóm tuổi
age_groups = pd.cut(data['age'], bins=[0, 40, 50, 60, data['age'].max()], labels=['<40 tuổi', '40-50', '50-60', 'Trên 60'])
# Vẽ box plot tỷ trọng mỡ theo giới tính và nhóm tuổi
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='pcfat', hue=age_groups, data=data)
plt.title('Box Plot tỷ trọng mỡ theo giới tính và nhóm tuổi')
plt.xlabel('Giới tính')
plt.ylabel('Tỷ trọng mỡ (pcfat)')
plt.legend(title='Nhóm tuổi')

# =====================c====================
# Vẽ histogram tỷ trọng mỡ theo giới tính
plt.figure(figsize=(8, 6))
sns.histplot(data, x='pcfat', hue='gender', common_norm=False, kde=True)
plt.title('Histogram tỷ trọng mỡ theo giới tính')
plt.xlabel('Tỷ trọng mỡ (pcfat)')

# =====================d====================
# Vẽ biểu đồ mối quan hệ giữa tuổi và mật độ xương theo giới tính
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='wbbmd', hue='gender', data=data)
plt.title('Mối quan hệ giữa tuổi và mật độ xương theo giới tính')
plt.xlabel('Tuổi')
plt.ylabel('Mật độ xương (wbbmd)')

# =====================e====================
# Vẽ biểu đồ mối quan hệ giữa BMI và tỷ trọng mỡ theo giới tính
plt.figure(figsize=(8, 6))
sns.scatterplot(x='bmi', y='pcfat', hue='gender', data=data)
plt.title('Mối quan hệ giữa chỉ số BMI và tỷ trọng mỡ theo giới tính')
plt.xlabel('Chỉ số BMI')
plt.ylabel('Tỷ trọng mỡ (pcfat)')
plt.show()

# Chỉ số BMI càng cao tỷ trọng mỡ càng lớn


