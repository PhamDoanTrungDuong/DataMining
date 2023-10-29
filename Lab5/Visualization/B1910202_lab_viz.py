#============================================1====================================================
## Bước #1: Import các thư viện cần thiết!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
"""## Bước #2: Đọc dữ liệu. gym_height.csv là dữ liệu chiều cao của 500
khách hàng ở một phòng tập gym """
df = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/gym_height.csv', index_col=0)

df.head()
"""## Bước #3: Vẽ histogram cho thuộc tính chiều cao!"""
# df.hist(column='height', grid=False)
# df.hist(column='height', grid=False, bins=20)

# ### Sinh viên so sánh sự khác biệt giữa 2 histogram và tìm hiểu thông số bins
# df.hist(column='height', by='sex', grid=False, figsize = (14,7))

"""## Một số yêu cầu """
### a. Vẽ histogram cho dữ liệu chiều cao của từng nhóm khách hàng (nam/nữ).Gợi ý: sử dụng thông số by.
# Vẽ histogram cho dữ liệu chiều cao của từng nhóm khách hàng (nam/nữ)
df.hist(column='height', by='sex', grid=False, bins=20)
plt.suptitle("Histogram chiều cao của nam và nữ")
plt.xlabel("Chiều cao")
plt.ylabel("Số lượng")
plt.show()

### b. Vẽ histogram chiều cao của nam và nữ trên cùng 1 biểu đồ.
# Tách dữ liệu theo giới tính
heights_male = df[df['sex'] == 'm']['height']
heights_female = df[df['sex'] == 'f']['height']

# Vẽ histogram cho chiều cao của nam và nữ trên cùng một biểu đồ
plt.hist(heights_male, bins=20, alpha=0.5, label='Nam', color='blue')
plt.hist(heights_female, bins=20, alpha=0.5, label='Nữ', color='pink')
plt.xlabel('Chiều cao')
plt.ylabel('Số lượng')
plt.title('Histogram chiều cao nam và nữ')
plt.legend()
plt.grid(False)
plt.show()


#============================================2====================================================
### Bước #1: Import các thư viện cần thiết
import seaborn as sns
import matplotlib.pyplot as plt
### Bước #2: Nạp bộ dữ liệu tips trong thư viện seaborn
tips = sns.load_dataset('tips')
tips.head()

## Bước #3a: vẽ biểu đồ hộp (Box Plot) cho thuộc tính total_bill
sns.boxplot(y=tips["total_bill"])
plt.title('Biểu đồ hộp của total_bill')
plt.xlabel('Total Bill')
plt.show()

# Vẽ một biểu đồ hình hộp cho thuộc tính total_bill được nhóm theo các ngàytrong tuần
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title('Biểu đồ hộp của total_bill theo day')
plt.xlabel('Total Bill By Day')
plt.show()

# Vẽ một biểu đồ hình hộp cho thuộc tính tổng hóa đơn của những khách hút thuốc và không hút thuốc được nhóm theo các ngày trong tuần
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips,
palette="Set3")
plt.title('Biểu đồ hộp của total_bill theo day')
plt.xlabel('Total Bill By Group by Smoker')
plt.show()


#============================================3====================================================
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


#============================================4====================================================
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


#============================================5====================================================
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

#============================================6====================================================
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

#============================================7====================================================
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

