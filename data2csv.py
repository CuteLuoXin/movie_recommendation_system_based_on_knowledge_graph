# D:\Dataset\ml-1m\ml-1m
import pandas as pd

# 读取数据集
df = pd.read_csv(f'D:\Dataset\ml-1m\ml-1m\\ratings.dat', sep='::', header=None,encoding='latin1', engine='python')

# 将数据集保存为CSV文件
df.to_csv(f"D:\Dataset\ml-1m\ml-1m\\ratings.csv", index=False, header=False)
