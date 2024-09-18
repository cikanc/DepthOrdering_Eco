import pandas as pd
import json

# 读取之前生成的 depth_summary.csv
depth_summary_path = "depth_summary.csv"
depth_summary = pd.read_csv(depth_summary_path)

# 读取原始注释文件
annotations_path = "train-annotations.json"
with open(annotations_path) as f:
    annotations = json.load(f)

# 创建一个字典，以便快速查找每个 image_id 对应的 Interdepth 和 Intradepth
image_id_to_depth = {}
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    attributes = annotation.get('attributes', {})

    interdepth = attributes.get('Interdepth')
    intradepth = attributes.get('Intradepth')

    # 只在存在时添加
    if interdepth is not None and intradepth is not None:
        image_id_to_depth[image_id] = {
            'Interdepth': interdepth,
            'Intradepth': intradepth
        }

    # 通过合并数据添加 Interdepth 和 Intradepth 字段
depth_summary['Interdepth'] = depth_summary['image_id'].map(
    lambda id: image_id_to_depth[id]['Interdepth'] if id in image_id_to_depth else None)
depth_summary['Intradepth'] = depth_summary['image_id'].map(
    lambda id: image_id_to_depth[id]['Intradepth'] if id in image_id_to_depth else None)

# 更新列顺序
depth_summary = depth_summary[['image_id', 'depth', 'Interdepth', 'Intradepth']]

# 添加 id 列（序号）
depth_summary.insert(0, 'id', range(1, len(depth_summary) + 1))

# 保存更新后的 CSV 文件
output_csv_path = "updated_depth_summary.csv"
depth_summary.to_csv(output_csv_path, index=False)

print(f"Updated CSV file created at: {output_csv_path}")