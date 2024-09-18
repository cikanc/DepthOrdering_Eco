import cv2
import numpy as np
import torch
from models.model import EcoDepth
from configs.infer_options import InferOptions
from utils import colorize_depth
import math
import os  # 导入 os 模块


def predict(orig_img, model, device):
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_img = orig_img / 255.0
    orig_h, orig_w, _ = orig_img.shape
    max_area = 1000 * 720
    area = orig_h * orig_w
    ratio = math.sqrt(area / max_area)

    new_h = int(orig_h / ratio)
    new_w = int(orig_w / ratio)
    new_img = cv2.resize(orig_img, (new_w, new_h))

    add_h = 64 - new_h % 64
    add_w = 64 - new_w % 64

    final_h = new_h + add_h
    final_w = new_w + add_w

    final_img = np.zeros((final_h, final_w, 3))
    final_img[:new_h, :new_w, :] = new_img

    final_img = torch.from_numpy(final_img)
    final_img = final_img.permute(2, 0, 1)
    final_img = final_img.unsqueeze(0)
    final_img = final_img.to(device)

    final_img_flipped = torch.flip(final_img, [3])
    final_img_concat = torch.cat([final_img, final_img_flipped])
    final_img_concat = final_img_concat.to(torch.float32)

    with torch.no_grad():
        final_depth_concat = model(final_img_concat)['pred_d']

    final_depth = final_depth_concat[0]
    final_depth_flipped = final_depth_concat[1]

    final_depth = (final_depth + torch.flip(final_depth_flipped, [2])) / 2

    final_depth = final_depth.squeeze()

    final_depth = final_depth[:new_h, :new_w]

    final_depth = final_depth.detach().cpu().numpy()

    final_depth = cv2.resize(final_depth, (orig_w, orig_h))

    return final_depth


def visualize_depth(depth):
    depth_map = colorize_depth(np.log(depth))
    depth_map = depth_map[:, :, ::-1]  # 假设colorize_depth返回的是BGR格式，这里转换为RGB
    return depth_map.astype(np.uint8)


def main():
    opt = InferOptions()
    args = opt.initialize().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EcoDepth(args=args)
    model_weight = torch.load(args.ckpt_dir)['model']
    model.load_state_dict(model_weight)
    model.to(device)
    model.eval()

    # 处理文件夹中的所有图像
    if args.img_path is not None:
        input_folder = args.img_path
        output_folder = 'infer_train'  # 输出文件夹
        os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

        print("Converting images in {} to depth maps".format(input_folder))
        for img_name in os.listdir(input_folder):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                img_path = os.path.join(input_folder, img_name)
                depth_name = img_name[:-4] + '_depth.png'
                depth_path = os.path.join(output_folder, depth_name)
                print(img_path)

                print("Converting {} to a depth map".format(img_path))
                img = cv2.imread(img_path)

                if img is not None:
                    depth = predict(img, model, device)
                    viz=visualize_depth(depth)

                    cv2.imwrite(depth_path, viz)
                    print("Saved depth map to {}".format(depth_path))
                else:
                    print("Error reading image {}".format(img_path))


if __name__ == '__main__':
    main()