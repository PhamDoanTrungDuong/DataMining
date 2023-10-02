# Biểu diễn tập dữ liệu lên mặt phẳng toạ độ Oxy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data_path = 'https://raw.githubusercontent.com/ltdaovn/dataset/master/housing2.csv'
data = pd.DataFrame(pd.read_csv(data_path))
data.describe()
data.info()

X = data[['Diện tích']]
y = data['Giá']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Xây dựng mô hình hồi quy với thư viện sklearn
import sklearn
from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X_train,y_train)

area = float(input("Nhập diện tích: "))

predicted_price = lm.predict([[area]])
print(f"${predicted_price[0]:,.2f}")


