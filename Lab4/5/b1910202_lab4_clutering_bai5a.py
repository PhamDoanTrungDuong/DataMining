import pandas as pd

# Đọc dữ liệu từ URL
url = "https://raw.githubusercontent.com/ltdaovn/dataset/master/Eurojobs.csv"
df = pd.read_csv(url)

# Tìm quốc gia có tỷ lệ dân làm trong ngành Tài chính (Finance) cao nhất
max_finance_country = df[df['Fin'] == df['Fin'].max()]['Country'].values[0]

print("Quốc gia có tỷ lệ dân làm trong ngành Tài chính (Finance) cao nhất là:", max_finance_country)
