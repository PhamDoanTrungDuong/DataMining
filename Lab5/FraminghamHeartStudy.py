# Bước 1. Nạp các thư viện cần thiết và đọc dữ liệu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
# Đọc tập tin dữ liệu
df = pd.read_csv(
'https://raw.githubusercontent.com/ltdaovn/dataset/master/framingham.csv')
# Xem các mẫu tin đầu của tập dữ liệu
df.head()
"""# Bước 2: Phân tích dữ liệu """
# Xem tỉ lệ nam/nữ trong tập dữ liệu
sns.countplot(x=df["male"]).set_title("Male/Female Ratio")
plt.show()
# Xem tỉ lệ bệnh, không bệnh trong dữ liêu
sns.countplot(x=df["TenYearCHD"]).set_title("Outcome Count")
plt.show()
# Xem tỉ lệ bệnh, không bệnh theo giới tính
sns.countplot(x="TenYearCHD", hue="male", data=df).set_title('Outcome Countby Gender')
plt.show()
"""# Bước #3: Làm sạch dữ liệu"""
# Số lượng dữ liệu bị thiếu trong các cột
missing_val_count_by_column = (df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# Xóa bỏ các dữ liệu rỗng
df = df.dropna()
print("df has null values: ", df.isnull().values.any())
"""# Bước #4: Cân bằng dữ liệu"""
from imblearn.under_sampling import RandomUnderSampler
X = df.drop(columns="TenYearCHD", axis=0)
y = df["TenYearCHD"]
rus = RandomUnderSampler(random_state=42)
df_data, df_target = rus.fit_resample(X, y)
# Xem lại dữ liệu sau khi đã được cân bằng
sns.countplot(df_target).set_title('Balanced Data Set')
plt.show()
"""# Bước #5: Xây dựng mô hình và đánh giá"""
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
scoring = {'accuracy':make_scorer(accuracy_score),
 'precision':make_scorer(precision_score),
 'recall':make_scorer(recall_score),
 'f1_score':make_scorer(f1_score)}
# Khởi tạo các mô hình
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
# Định nghĩa hàm đánh giá
def models_evaluation(X, y, folds):

 '''
 X : data set features
 y : data set target
 folds : number of cross-validation folds

 '''

 # Tính cross-validation cho từng mô hình
 log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
 svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
 dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
 rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
 gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
 models_scores_table = pd.DataFrame(
{'Logistic Regression':[log['test_accuracy'].mean(),
log['test_precision'].mean(),
 log['test_recall'].mean(),
 log['test_f1_score'].mean()],
 'Support Vector Classifier':[svc['test_accuracy'].mean(),
svc['test_precision'].mean(),
 svc['test_recall'].mean(),
 svc['test_f1_score'].mean()],

 'Decision Tree':[dtr['test_accuracy'].mean(),
 dtr['test_precision'].mean(),
 dtr['test_recall'].mean(),
 dtr['test_f1_score'].mean()],

 'Random Forest':[rfc['test_accuracy'].mean(),
rfc['test_precision'].mean(),
 rfc['test_recall'].mean(),
 rfc['test_f1_score'].mean()],

 'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
gnb['test_precision'].mean(),
 gnb['test_recall'].mean(),
 gnb['test_f1_score'].mean()]},

 index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

 # Thêm cột 'Best Score'
 models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
 return(models_scores_table)

# Thực thi mô hình và đánh giá các mô hình
kfolds = 5
print(models_evaluation(df_data, df_target, kfolds))