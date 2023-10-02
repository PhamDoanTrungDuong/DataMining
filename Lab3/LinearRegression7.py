import matplotlib.pyplot as plt

# Dữ liệu
conc = [1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
strength = [6.3, 11.1, 20.0, 24.0, 26.1, 30.0, 33.8, 34.0, 38.1, 39.9, 42.0, 46.1, 53.1, 52.0, 52.5, 48.0, 42.8, 27.8, 21.9]

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.scatter(conc, strength, c='b', marker='o', label='Data Points')
plt.title('Mối liên hệ giữa hàm lượng gỗ cứng và độ căng mạnh')
plt.xlabel('Hàm lượng gỗ cứng')
plt.ylabel('Độ căng mạnh')
plt.grid(True)
plt.legend()
plt.show()


from sklearn.linear_model import LinearRegression
import numpy as np

# Biến đổi dữ liệu thành mảng numpy
conc = np.array(conc).reshape(-1, 1)
strength = np.array(strength)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(conc, strength)

# Lấy hệ số và sai số của mô hình
slope = model.coef_[0]
intercept = model.intercept_

# In phương trình mô hình
print(f'Phương trình mô hình: Độ căng mạnh = {slope:.2f} * Hàm lượng gỗ cứng + {intercept:.2f}')
