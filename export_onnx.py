import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

# --- 配置 ---
DEVICE = 'cpu'
# 选择要导出的模型: 'vits', 'vitb', 'vitl', 或 'vitg'
ENCODER_TO_EXPORT = 'vitl'
# 样例图片路径
IMAGE_PATH = './assets/examples/demo01.jpg'
# 导出的模型文件基础路径
EXPORT_PATH_BASE = f'./onnx/depth_anything_v2_{ENCODER_TO_EXPORT}'

# --- 模型定义 ---
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def export_depth_anything_v2(encoder_type='vits', device='cpu', export_path_base=None, image_path=None, is_fp16=True):
    """
    导出 DepthAnythingV2 模型到 PNNX/NCNN 格式，并进行可视化比较

    Args:
        encoder_type: 模型编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
        device: 运行推理的设备 ('cpu' 或 'cuda')
        export_path_base: 导出的模型文件的基础路径
        image_path: 用于可视化比较的样例图片路径
    """
    if export_path_base is None:
        export_path_base = f'./onnx/depth_anything_v2_{encoder_type}'

    # --- 1. 加载 PyTorch 模型 ---
    print(f"正在加载 PyTorch 模型: depth_anything_v2_{encoder_type}")
    model = DepthAnythingV2(**model_configs[encoder_type])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    model = model.to(device).eval()

    # --- 2. 导出模型到 ONNX ---
    print("\n正在导出模型到 ONNX 格式...")
    # 使用两个尺寸为 14 的倍数的不同随机输入来处理动态形状
    # 例如, 518 = 14 * 37, 392 = 14 * 28, 644 = 14 * 46
    torch.onnx.export(model, torch.rand(1, 3, 532, 532).to(device),f'{export_path_base}.onnx',
                      input_names=['input'],
                      output_names=['output'],)

    print(f"模型成功导出到: {export_path_base}.onnx")



# --- 主执行程序 ---
if __name__ == "__main__":
    # 导出指定的模型
    export_depth_anything_v2(
        encoder_type=ENCODER_TO_EXPORT,
        device=DEVICE,
        export_path_base=EXPORT_PATH_BASE,
        image_path=IMAGE_PATH,
        is_fp16=True
    )

    # 示例: 导出多个模型
    # for encoder in ['vits', 'vitb', 'vitl']:
    #     export_depth_anything_v2(encoder_type=encoder, image_path=IMAGE_PATH)