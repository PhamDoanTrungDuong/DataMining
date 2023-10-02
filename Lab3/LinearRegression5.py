import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
age = np.array([46, 20, 52, 30, 57, 25, 28, 36, 22, 43, 57, 33, 22, 63, 40, 48, 28, 49])
bmi = np.array([25.4, 20.6, 26.2, 22.6, 25.4, 23.1, 22.7, 24.9, 19.8, 25.3, 23.2, 21.8, 20.9, 26.7, 26.4, 21.2, 21.2, 22.8])
chol = np.array([3.5, 1.9, 4.0, 2.6, 4.5, 3.0, 2.9, 3.8, 2.1, 3.8, 4.1, 3.0, 2.5, 4.6, 3.2, 4.2, 2.3, 4.0])

# Quan hệ giữa Age và Chol
plt.scatter(age, chol, color='blue', label='Data Points')
plt.title('Quan hệ giữa Age and Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.grid(True)

model_age = LinearRegression()
model_age.fit(age.reshape(-1, 1), chol)

age_range = np.linspace(min(age), max(age), 100).reshape(-1, 1)
predicted_chol_age = model_age.predict(age_range)

plt.plot(age_range, predicted_chol_age, color='red', linewidth=2, label='Regression Line')

plt.legend()
plt.show()

# Quan hệ giữa Age, BMI, Chol
plt.scatter(age, chol, c=bmi, cmap='viridis', label='Data Points')
plt.title('Quan hệ giữa Age, BMI, and Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.colorbar(label='BMI')
plt.grid(True)

X = np.column_stack((age, bmi))
model_age_bmi = LinearRegression()
model_age_bmi.fit(X, chol)

predicted_chol_age_bmi = model_age_bmi.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(age, bmi, chol, c='b', marker='o', label='Data Points')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Cholesterol')

age_range, bmi_range = np.meshgrid(age_range, np.linspace(min(bmi), max(bmi), 100))
X_pred = np.column_stack((age_range.ravel(), bmi_range.ravel()))

predicted_chol_surface = model_age_bmi.predict(X_pred)
predicted_chol_surface = predicted_chol_surface.reshape(age_range.shape)

ax.plot_surface(age_range, bmi_range, predicted_chol_surface, cmap='viridis', alpha=0.5, label='Regression Surface')

plt.show()
