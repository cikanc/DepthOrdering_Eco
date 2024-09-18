import pandas as pd

# 读取CSV文件
depth_train = pd.read_csv('depth_train.csv')

# 删除pred_Intradepth或pred_Interdepth字段值为空的行
filtered_df = depth_train.dropna(subset=['pred_Intradepth', 'pred_Interdepth'])

# 保存为新的CSV文件
filtered_df.to_csv('depth_train_unique.csv', index=False)