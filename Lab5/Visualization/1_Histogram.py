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