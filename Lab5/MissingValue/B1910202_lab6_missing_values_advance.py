# -*- coding: utf-8 -*-

"""1. Bài tập về xử lý các trường hợp dữ liệu bị thiếu

Trong bài tập này, chúng ta sẽ làm việc với bộ dữ liệu Housing Prices. 

Tập dữ liệu này có 79 giá trị mô tả (gần như) mọi khía cạnh của các căn nhà ở tại Ames, Iowa.
Với tập dữ liệu này, chúng ta cần xây dựng mô hình để dự đoán giá của mỗi ngôi nhà.

"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
X_full = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/housing-prices/train.csv',
                     index_col='Id')
X_test_full = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/housing-prices/test.csv',
                          index_col='Id')


# Loại bỏ các căn nhà không có giá
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Để đơn giản bài toán, ở đây chúng ta chỉ chọn các thuộc tính số
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

X.fillna(X.mean(), inplace=True)

# Chia tập dữ liệu thành 2 tập dữ liệu con là training set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)

# # Xem các căn nhà đầu tiên trong tập dữ liệu huấn luyện. Chú ý các giá trị bị thiếu.
# X_train.info()

# total_missing_entries = X_train.isnull().sum().sum()
# print("Total number of missing entries in the training data:", total_missing_entries)


# """
# # Bước 1: Làm quen với dữ liệu
# """

# # Xem mô tả tập dữ liệu huấn luyện
print(X_train.shape)

# Số lượng dữ liệu bị thiếu trong các cột
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Hãy trả lời các câu hỏi sau đây?

# Có bao nhiêu căn nhà trong tập dữ liệu huấn luyện?
# num_rows = 1168 

# Có bao nhiêu cột dữ liệu bị thiếu?
# num_cols_with_missing = 3

# How many missing entries are contained in all of the training data?
# tot_missing = 276

# Theo bạn, phương pháp nào thích hợp nhất để xử lý trường hợp bị thiếu dữ liệu này?


# """Bước 2: định nghĩa hàm để đo chất lượng của từng phương pháp

# Để so sánh chất lượng của các phương pháp, chúng ta cần định nghĩa hàm score_dataset() . 
# Hàm được sử dụng trong ví dụ này là hàm Trung bình của sai biệt tuyệt đối (the mean absolute error (MAE)) 
# dành cho mô hình rừng ngẫu nhiên (RandomForest).
# (https://en.wikipedia.org/wiki/Mean_absolute_error) 
# """


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor(n_estimators=100, random_state=0)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

result = score_dataset(X_train, X_valid, y_train, y_valid)

print(result)

# """
# # Bước 3: 

# Trong các phương pháp xử lý dữ liệu bị thiếu bạn đã học,
# phương pháp nào cho kết quả dự báo chính xác nhất.
# Hãy cho biết giá trị MAE của từng mô hình.

# Mô hình Random Forest trước khi xử lý dữ liệu bị thiếu
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
mae_before = mean_absolute_error(y_valid, preds)

# Xử lý giá trị bị thiếu bằng cách điền giá trị trung bình
X_filled_mean = X.fillna(X.mean())
X_train_filled_mean = X_train.fillna(X_train.mean())
X_valid_filled_mean = X_valid.fillna(X_train.mean())

model.fit(X_train_filled_mean, y_train)
preds = model.predict(X_valid_filled_mean)
mae_mean = mean_absolute_error(y_valid, preds)

# Xử lý giá trị bị thiếu bằng cách sử dụng mô hình học máy
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
mae_model = mean_absolute_error(y_valid, preds)

print("MAE trước khi xử lý dữ liệu bị thiếu:", mae_before)
print("MAE sau khi điền giá trị trung bình:", mae_mean)
print("MAE sau khi sử dụng mô hình học máy:", mae_model)

# """
