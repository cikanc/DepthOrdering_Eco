import json
import os
from PIL import Image
import cv2

# 定义文件路径
annotations_file = '/tmp/EcoDepth/depth_test.json'
infer_train_crop_dir = '/tmp/EcoDepth/depth/infer_test_crop/'
output_dir = 'infer_test1/'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 加载annotations
with open(annotations_file, 'r') as f:
    data = json.load(f)

annotations = data['annotations']
images = {img['id']: img for img in data['images']}  # 以id为键获取每个image的详细信息

# 按照image_id分组annotations
image_annotations = {}
for ann in annotations:
    if len(ann['segmentation']) == 0 and ann['category_id'] == 25:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)

# 合并bounding box并保存图片
for image_id, ann_list in image_annotations.items():
    if image_id in images:
        image_info = images[image_id]
        filename = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        filename_without_ext, ext = os.path.splitext(filename)
        filename = filename_without_ext + "_depth" + ext

        # 创建合成图片，使用原始图片的尺寸
        output_image = cv2.imread("/tmp/EcoDepth/depth/infer_test/"+filename)
        output_image = Image.fromarray(output_image)
        # output_image = Image.new('RGB', (width, height))

        for ann in ann_list:
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            crop_image_path = os.path.join(infer_train_crop_dir, f"{ann['id']}+{filename}")

            # 读取裁剪后的图片
            try:
                crop_image = Image.open(crop_image_path)
                output_image.paste(crop_image, (x, y))
            except FileNotFoundError:
                print(f"File not found: {crop_image_path}")
        # 保存合成后的图片
        output_image_path = os.path.join(output_dir, filename)
        output_image.save(output_image_path)
        print(f"Saved combined image: {output_image_path}")