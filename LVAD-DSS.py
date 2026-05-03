import random
import os
from typing import Optional, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt

def simulate_low_light(input_image_path: str, output_image_path: str,
                       fixed_C5: Optional[int] = None,
                       alpha: Optional[float]= None , beta: Optional[float]= None, gamma: Optional[float]= None) -> None:
    """
    使用给定参数将图像转换为低光版本，并添加混合噪声。
    """
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"无法读取输入图像: {input_image_path}")

    image_norm = image.astype(np.float32) / 255.0
    low_light_image = np.zeros_like(image_norm)

    for i in range(3):
        low_light_image[:, :, i] = beta * (alpha * image_norm[:, :, i]) ** gamma

    # 添加混合噪声
    #low_light_image = add_strong_mixed_noise(image_norm)
    low_light_image = add_sensor_noise(low_light_image, fixed_C5)

    # 保存图像
    low_light_image = (low_light_image * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, low_light_image)

    # # 可视化
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # axs[0].imshow(image_norm)
    # axs[0].set_title("加噪前（低光模拟）")
    # axs[0].axis('off')
    #
    # axs[1].imshow(low_light_image)
    # axs[1].set_title("加噪后（真实噪声）")
    # axs[1].axis('off')
    #
    # plt.tight_layout()
    # plt.show()


def add_sensor_noise(image: np.ndarray, fixed_C5: int = None) -> np.ndarray:
    """
    添加符合 Brooks et al. (2019) 和 Foi et al. (2008) 的真实相机传感器噪声。
    模拟低光条件下，包含 Poisson（shot）+ Gaussian（read）噪声。

    参数:
        image: 输入图像 (float32, 已归一化到[0,1])
        fixed_C5: 可选的噪声等级（0-19），如为 None 则随机选择

    返回:
        添加噪声后的图像，仍在[0,1]范围内
    """
    #assert image.dtype == np.float32 and image.max() <= 1.0, "图像必须为 float32 且归一化到[0,1]"

    # 生成或固定噪声等级 C5 ∈ {0, 1, ..., 19}
    C5 = random.randint(0, 19) if fixed_C5 is None else fixed_C5

    # log-uniform 分布生成 shot noise 参数 λ_shot
    log_lambda_shot = random.uniform(np.log(1e-6 + 1e-6 * C5),
                                     np.log(7e-6 + 7e-6 * (C5 + 1)))
    lambda_shot = np.exp(log_lambda_shot)

    # log-linear 生成 read noise 参数 λ_read（单位高斯）
    log_lambda_read = 2.18 * np.log(lambda_shot) + 1.2 + np.random.normal(0, 0.26)
    lambda_read = np.exp(log_lambda_read)

    # 计算 signal-dependent σ
    sigma = np.sqrt(lambda_read + lambda_shot * image)

    # 添加高斯噪声
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def generate_parameters(fixed_C5: Optional[int], alpha: Optional[float], beta: Optional[float], gamma: Optional[float]):
    """
    返回指定或随机生成的 alpha, beta, gamma 参数
    """
    if fixed_C5 is None:
        fixed_C5 = random.randint(0, 19)
    if alpha is None:
        alpha = random.uniform(0.9, 1.0)
    if beta is None:
        beta = random.uniform(0.5, 1.0)
    if gamma is None:
        gamma = random.uniform(1.5, 5.0)
    return fixed_C5, alpha, beta, gamma


def process_single_image(image_path: str, output_path: str,
                         fixed_C5: Optional[int],
                         alpha: Optional[float] = None,
                         beta: Optional[float] = None,
                         gamma: Optional[float] = None) -> None:
    fixed_C5, alpha, beta, gamma = generate_parameters(fixed_C5, alpha, beta, gamma)
    print(f"处理单张图片参数:fixed_C5={fixed_C5:.2f}, alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")
    simulate_low_light(image_path, output_path, fixed_C5, alpha, beta, gamma)
    print(f"成功处理: {image_path}")


def process_folder(input_dir: str, output_dir: str,
                   fixed_C5: Optional[int] = None,
                   alpha: Optional[float] = None,
                   beta: Optional[float] = None,
                   gamma: Optional[float] = None) -> None:
    """
    如果指定了 fixed_C5/alpha/beta/gamma，则文件夹内所有图片使用同一组参数；
    如果没指定，则每张图片独立生成随机参数。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 判断是否为“固定参数模式”
    fixed_params_mode = all(p is not None for p in [fixed_C5, alpha, beta, gamma])
    if fixed_params_mode:
        print(f"固定参数模式: fixed_C5={fixed_C5}, alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('medium.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)

            # 每张图生成参数（如果是随机模式）
            if fixed_params_mode:
                sub_C5, sub_alpha, sub_beta, sub_gamma = fixed_C5, alpha, beta, gamma
            else:
                sub_C5, sub_alpha, sub_beta, sub_gamma = generate_parameters(fixed_C5, alpha, beta, gamma)

            print(f"处理图片: {filename} -> 参数: C5={sub_C5}, alpha={sub_alpha:.2f}, beta={sub_beta:.2f}, gamma={sub_gamma:.2f}")

            output_path = os.path.join(output_dir, filename)
            try:
                simulate_low_light(input_path, output_path, sub_C5, sub_alpha, sub_beta, sub_gamma)
                print(f"成功处理: {filename}")
            except Exception as e:
                print(f"处理失败: {filename}, 错误: {str(e)}")


def batch_process(root_input_dir: str, root_output_dir: str,
                  fixed_C5: Optional[int] = None,
                  alpha: Optional[float] = None,
                  beta: Optional[float] = None,
                  gamma: Optional[float] = None) -> None:
    """
    处理嵌套文件夹结构，每个子文件夹内的所有图片共享同一个 C5（除非手动指定 fixed_C5）。
    """
    for foldername in os.listdir(root_input_dir):
        input_folder = os.path.join(root_input_dir, foldername)
        output_folder = os.path.join(root_output_dir, foldername)

        if os.path.isdir(input_folder):
            # 如果没手动指定 fixed_C5，就为当前文件夹生成一个随机 C5
            folder_C5 = fixed_C5 if fixed_C5 is not None else random.randint(0, 19)
            print(f"\n处理文件夹: {foldername}，C5={folder_C5}")

            process_folder(input_folder, output_folder,
                           fixed_C5=folder_C5,
                           alpha=alpha,
                           beta=beta,
                           gamma=gamma)

def process_folder_suiji(input_dir: str, output_dir: str,
                   fixed_C5: Optional[int] = None,
                   alpha: Optional[float] = None,
                   beta: Optional[float] = None,
                   gamma: Optional[float] = None,
                   ) -> None:
    """
    对文件夹中每一张图片使用独立的随机参数进行低光+噪声处理，并打印参数。
    处理后图像命名中包含参数信息，方便溯源。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('medium.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)

            # 每张图独立生成参数
            sub_C5, sub_alpha, sub_beta, sub_gamma = generate_parameters(fixed_C5, alpha, beta, gamma)
            print(f"处理图片: {filename} -> 参数: C5={sub_C5}, alpha={sub_alpha:.2f}, beta={sub_beta:.2f}, gamma={sub_gamma:.2f}")

            # 构造带参数信息的输出文件名
            name, ext = os.path.splitext(filename)
            param_str = f"c{sub_C5}_a{sub_alpha:.2f}_b{sub_beta:.2f}_g{sub_gamma:.2f}"
            output_filename = f"{name}_{param_str}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            try:
                simulate_low_light(input_path, output_path, sub_C5, sub_alpha, sub_beta, sub_gamma)
                print(f"成功处理: {filename}")
            except Exception as e:
                print(f"处理失败: {filename}, 错误: {str(e)}")

# 示例使用 1：处理单张图片（自动生成参数）
#process_single_image(r"E:\data\ShanghaiTechDataset\training\train_frames\01_001\frame_00000.jpg", r"C:\Users\hony\Desktop\000000.jpg",fixed_C5=None, alpha=0.62, beta=0.62, gamma=2.12)
# process_folder_suiji(r"E:\data\ShanghaiTechDataset\training\train_frames\01_052",
#                r"C:\Users\hony\Desktop\01_052shishi0",
#                fixed_C5=None, alpha=None, beta=None, gamma=None)
# 示例使用 2：处理单个文件夹（指定参数）
process_folder(r"/home/zhouhao/PycharmProjects/lxhanProject/aed-mae/data/avenue/test/frames-ori/04", r"/home/zhouhao/PycharmProjects/lxhanProject/frames",fixed_C5=None, alpha=0.77, beta=1.92, gamma=2.18)

# 示例使用 3：处理嵌套文件夹（每个子文件夹独立参数）
# batch_process("/media/lmy/processing/han/aed-mae-xin/data/ShanghaiTech_0.93_0.71_1.66/test/frames", "/media/lmy/processing/han/aed-mae-xin/data/ShanghaiTech_0.93_0.71_1.66/test/frames_",
#               alpha=0.93,beta=0.71,gamma=1.66)

# 示例使用 4：处理嵌套文件夹（所有子文件夹统一使用指定参数）
#batch_process(r"/home/zhouhao/PycharmProjects/lxhanProject/aed-mae/data/avenue/test/frames", r"/home/zhouhao/PycharmProjects/lxhanProject/frames",fixed_C5=None, alpha=0.62, beta=0.62, gamma=2.12)
#process_folder_suiji('/media/lmy/processing/han/aed-mae-xin/data/avenue/train/frames/01','/media/lmy/processing/han/SCI/data/01_random')