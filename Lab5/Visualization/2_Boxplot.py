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
