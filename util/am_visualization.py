import os
import numpy as np
import cv2


def read_images(root_dir, image_name, image_size=224, border_ratio=0., crop_window=(0, 0, 224, 224)):
    def crop_image(image, crop_window):
        _image = image[crop_window[0]:crop_window[2], crop_window[1]:crop_window[3], :]
        return _image

    gt_path = os.path.join(root_dir, image_name + '_mask.png')
    ori_path = os.path.join(root_dir, image_name + '_img.png')
    img_path = os.path.join(root_dir, image_name + f'_amp.png')

    if not crop_window:
        crop_window = (0, 0, image_size, image_size)

    print(f'gt path:{gt_path}')
    print(f'ori path:{ori_path}')
    print(f'score path:{img_path}')

    try:
        gt_image = cv2.imread(gt_path)
        ori_image = cv2.imread(ori_path)
        img = cv2.imread(img_path)

        gt_image_crop = crop_image(gt_image, crop_window)
        ori_image_crop = crop_image(ori_image, crop_window)
        img_crop = crop_image(img, crop_window)

        gt_image = cv2.resize(gt_image, (image_size, image_size))
        ori_image = cv2.resize(ori_image, (image_size, image_size))
        img = cv2.resize(img, (image_size, image_size))

        gt_image_crop = cv2.resize(gt_image_crop, (image_size, image_size))
        ori_image_crop = cv2.resize(ori_image_crop, (image_size, image_size))
        img_crop = cv2.resize(img_crop, (image_size, image_size))

    except:
        print("The path does not exist..")
        raise ValueError

    if border_ratio > 0:
        border_size = int(image_size * border_ratio)
        gt_image = add_black_border(gt_image, border_size)
        ori_image = add_black_border(ori_image, border_size)
        img = add_black_border(img, border_size)

    return ori_image, gt_image, img, ori_image_crop, gt_image_crop, img_crop


def add_black_border(image, border_size):
    assert len(image.shape) == 3
    image[:border_size, :, :] = 0
    image[-1 - border_size:-1, :, :] = 0
    image[:, :border_size, :] = 0
    image[:, -1 - border_size:-1, :] = 0

    return image


def concat_images(image_pairs, nrows, gap_ratio=0.1, save_path=None):
    # image_pairs: (ori_list, gt_list, score_list1.... score_listn)

    ori_image_list = image_pairs[0]

    ncols = ((len(ori_image_list) + nrows - 1) // nrows) * len(image_pairs)

    height = ori_image_list[0].shape[0]
    width = ori_image_list[0].shape[1]
    gap_size = int(height * gap_ratio)

    result_width = width * nrows + gap_size * (nrows - 1)
    result_height = height * ncols + gap_size * (ncols - 1)

    result_image = np.ones((result_height, result_width, 3), dtype=int) * 255

    for i in range(len(image_pairs)):
        for j in range(len(ori_image_list)):

            cur_row = j % nrows
            cur_col = (j // nrows) * len(image_pairs) + i

            if cur_row == 0:
                begin_w = 0
            else:
                begin_w = cur_row * (width + gap_size)

            if cur_col == 0:
                begin_h = 0
            else:
                begin_h = cur_col * (height + gap_size)

            result_image[begin_h:begin_h + height, begin_w:begin_w + width, :] = image_pairs[i][j]

    if save_path is None:
        cv2.imwrite("result_image.png", result_image)
    else:
        cv2.imwrite(save_path, result_image)


def plot_qualitative_results(root_dir, image_names, method_dirs, save_dir, save_name,
                             border_ratio=0, gap_ratio=0.07, image_size=224, crop_window_list=[],
                             reverse=False):
    os.makedirs(save_dir, exist_ok=True)
    all_category_img = None

    to_crop = True
    if not crop_window_list or len(crop_window_list) == 0:
        crop_window_list = [None for _ in image_names]
        to_crop = False

    assert len(image_names) == len(crop_window_list)

    for image_name, crop_window in zip(image_names, crop_window_list):

        category_imgs = None
        if to_crop:
            crop_imgs = None
        for idx in range(0, len(method_dirs)):
            subset_dir = method_dirs[idx]
            full_dir = os.path.join(root_dir, subset_dir)

            img_list = read_images(full_dir, image_name, image_size, border_ratio, crop_window)
            if category_imgs is None:
                category_imgs = [img_list[0], img_list[1]]  # ori + gt
                if to_crop:
                    crop_imgs = [img_list[3], img_list[4]]  # cropped ori + gt

            category_imgs.append(img_list[2])  # scores
            if to_crop:
                crop_imgs.append(img_list[5])  # scores

        if not reverse:
            if all_category_img is None:
                all_category_img = [[] for _ in category_imgs]

            for i in range(len(all_category_img)):
                all_category_img[i].append(category_imgs[i])
                if to_crop:
                    all_category_img[i].append(crop_imgs[i])
        else:
            if all_category_img is None:
                all_category_img = []
            all_category_img.append(category_imgs)
            if to_crop:
                all_category_img.append(crop_imgs)

    if to_crop:
        if not reverse:
            concat_images(all_category_img, 2 * len(image_names), gap_ratio,
                          save_path=os.path.join(save_dir, f'{save_name}.png'))
        else:
            concat_images(all_category_img, len(method_dirs) + 2, gap_ratio,
                          save_path=os.path.join(save_dir, f'{save_name}.png'))
    else:
        if not reverse:
            concat_images(all_category_img, len(image_names), gap_ratio,
                          save_path=os.path.join(save_dir, f'{save_name}.png'))
        else:
            concat_images(all_category_img, len(method_dirs) + 2, gap_ratio,
                          save_path=os.path.join(save_dir, f'{save_name}.png'))

import glob

def get_image_names_from_path(data_path):
    """
    从给定路径中提取所有图片的文件名前缀（不包含_img.png等后缀）
    例如：检测到xxx_img.png、xxx_mask.png、xxx_amp.png → 返回xxx
    """
    # 获取所有_img.png文件
    img_files = glob.glob(os.path.join(data_path, "*_img.png"))

    # 提取基础文件名（去掉_img.png后缀）
    image_names = [os.path.basename(f).replace("_img.png", "") for f in img_files]

    # 验证是否存在对应的mask和amp文件
    valid_names = []
    for name in image_names:
        mask_path = os.path.join(data_path, f"{name}_mask.png")
        amp_path = os.path.join(data_path, f"{name}_amp.png")
        if os.path.exists(mask_path) and os.path.exists(amp_path):
            valid_names.append(name)
        else:
            print(f"警告：文件{name}缺少配套的mask或amp图片")

    return valid_names


########### 绘制原始图 mask 异常分值图（可以有多个方法）的可视化图像
"""
root_dir/  # 存放所有方法目录的根目录  
├─ method_dir_1/  # 方法1（或参数1）的结果目录  
│  ├─ sample1_img.png  
│  ├─ sample1_mask.png  
│  ├─ sample1_amp.png  
│  ├─ sample2_img.png  
│  ├─ sample2_mask.png  
│  └─ sample2_amp.png  
├─ method_dir_2/  # 方法2（或参数2）的结果目录  
│  ├─ sample1_img.png  # 与method_dir_1的样本名一致（同一样本）  
│  ├─ sample1_mask.png  
│  ├─ sample1_amp.png  
│  └─ ...  
└─ ...  
"""

# 任选一个方法目录（用于提取有效样本名，需包含完整的 3 类文件），例如method_dir_1的路径。
data_path = "/home/user/zyx/HUST-ADer4YX_FUIAD/runs/DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores/dinomaly"

# 自动获取所有有效图片名
image_names = get_image_names_from_path(data_path)
print(f"检测到的图片名称列表: {image_names}")

# 调用可视化函数（示例）
if image_names:  # 确保有有效图片才执行
    plot_qualitative_results(
        root_dir="/home/user/zyx/HUST-ADer4YX_FUIAD/runs",
        image_names=image_names,
        method_dirs=["DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores/dinomaly",
                     "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores/dinomaly",
                     "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores/dinomaly"],
        save_dir="/home/user/zyx/HUST-ADer4YX_FUIAD",
        save_name="Anomaly_maps_visualization",
        image_size=224
    )
else:
    print("错误：未检测到有效的图片文件！")