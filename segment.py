"""裁剪bbox框并记录信息"""

import json
import os

from PIL import Image
from pycocotools.coco import COCO
import tqdm


def crop_images_and_create_json(bbox_folder, image_folder, output_folder):
    data = {}

    coco = COCO(bbox_folder)
    imgIds = coco.getImgIds()
    print(f"the length is {len(imgIds)}")

    k = 1
    for image_id in imgIds:
        img = coco.loadImgs(image_id)[0]
        img_name = img["file_name"]
        image_path = image_folder + img_name
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping...")
            continue

        image_pic = Image.open(image_path)

        annIds = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annIds)

        for i in tqdm.trange(len(anns), desc=f"this is the {k}th image"):
            ann = anns[i]
            if len(ann["segmentation"]) == 0 and ann["category_id"] == 25:
            # if ann['attributes'].get('Intradepth'):
                new_name = str(ann["id"]) + '+' + img_name
                new_path = output_folder + new_name
                [x, y, w, h] = ann["bbox"]
                cropped_img = image_pic.crop((x, y, x + w, y + h))
                cropped_img.save(new_path)
                data[new_name] = {"original_image": img_name,
                                  "id": ann["id"],
                                  "image_id": ann["image_id"],
                                  "category_id": ann["category_id"]}
        k += 1

    # json_path = 'image_bbox_relationships_train.json'
    json_path = 'image_bbox_relationships_test.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    # bbox_folder = "/tmp/EcoDepth/train-annotations.json"
    # image_folder = "/tmp/EcoDepth/train/"
    # output_folder = "/tmp/EcoDepth/train_crop/"
    bbox_folder = "/tmp/EcoDepth/depth_test.json"
    image_folder = "/tmp/EcoDepth/depth_test/"
    output_folder = "/tmp/EcoDepth/test_crop/"
    os.makedirs(output_folder, exist_ok=True)
    crop_images_and_create_json(bbox_folder, image_folder, output_folder)
