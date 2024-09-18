import os
import json
import numpy as np
import pandas as pd
from PIL import Image


def calculate_depth_in_bbox(depth_image, bbox, width, height):
    # 提取bbox坐标
    x, y, bbox_width, bbox_height = map(int, bbox)
    # x = x + width  # 根据您的需求，调整x坐标
    # 确保bbox在图像内部
    if x + bbox_width > width or y + bbox_height > height:
        print(f"Warning: BBox {bbox} is out of the image boundaries.")
        return 0

    depth_region = depth_image[y:y + bbox_height, x:x + bbox_width]
    return np.sum(depth_region)  # 计算深度和


def create_csv_from_annotations(annotations_path, output_csv_path):
    # 读取注释文件
    with open(annotations_path) as f:
        annotations = json.load(f)

        # 创建一个映射以根据image_id获取file_name, width和height
    image_id_to_info = {
        img['id']: (img['file_name'], img['width'], img['height'])
        for img in annotations['images']
    }

    data = []
    index = 0
    # 遍历所有注释
    for annotation in annotations['annotations']:
        print(index)
        index += 1
        annotation_id = annotation['id']
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        attributes = annotation.get('attributes', {})
        interdepth = attributes.get('Interdepth')
        intradepth = attributes.get('Intradepth')

        # 根据image_id获取对应的file_name, width和height
        if image_id in image_id_to_info:
            file_name, width, height = image_id_to_info[image_id]

            # 根据原始文件名生成深度图文件名
            depth_file_name = file_name.replace('.', '_depth.')
            depth_image_path = os.path.join("/tmp/EcoDepth/depth/infer_train/", depth_file_name)

            if os.path.exists(depth_image_path):
                depth_image = np.array(Image.open(depth_image_path))

                # 计算该bbox对应的深度和
                depth_sum = calculate_depth_in_bbox(depth_image, bbox, width, height)

                data.append({
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "pred_Intradepth": intradepth,
                    "pred_Interdepth": interdepth,
                    "depth": depth_sum
                })

                # 创建DataFrame并保存为csv
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)


# 设置路径
annotations_path = "./train-annotations.json"
output_csv_path = "./depth_train.csv"

# 创建CSV文件
create_csv_from_annotations(annotations_path, output_csv_path)

print(f"CSV file created at: {output_csv_path}")