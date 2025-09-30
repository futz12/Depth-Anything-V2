import cv2
import torch
import pnnx
import ncnn
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
EXPORT_PATH_BASE = f'./pnnx/depth_anything_v2_{ENCODER_TO_EXPORT}'

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
        export_path_base = f'./pnnx/depth_anything_v2_{encoder_type}'

    # --- 1. 加载 PyTorch 模型 ---
    print(f"正在加载 PyTorch 模型: depth_anything_v2_{encoder_type}")
    model = DepthAnythingV2(**model_configs[encoder_type])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    model = model.to(device).eval()

    # --- 2. 导出模型到 PNNX/NCNN ---
    print("\n正在导出模型到 PNNX/NCNN 格式...")
    # 使用两个尺寸为 14 的倍数的不同随机输入来处理动态形状
    # 例如, 518 = 14 * 37, 392 = 14 * 28, 644 = 14 * 46
    pnnx.export(model, f'{export_path_base}.pt',
                inputs=torch.rand(1, 3, 532, 532).to(device),
                inputs2=torch.rand(1, 3, 392, 644).to(device),
                fp16=is_fp16)
    print(f"模型成功导出到: {export_path_base}.param 和 {export_path_base}.bin")

    # --- 3. 使用样例图片进行可视化比较 ---
    if image_path:
        print("\n正在处理样例图片以进行可视化比较...")
        try:
            # 读取样例图片
            raw_img = cv2.imread(image_path)
            if raw_img is None:
                print(f"错误: 无法从 {image_path} 加载图片")
                return

            # PyTorch 推理
            print("正在运行 PyTorch 推理...")
            depth_pytorch = model.infer_image(raw_img)

            # NCNN 推理
            print("正在运行 NCNN 推理...")
            h, w = raw_img.shape[:2]
            # NCNN 的输入必须与 PyTorch 模型的 `image2tensor` 方法进行相同的预处理
            in0, (processed_h, processed_w) = model.image2tensor(raw_img)

            with ncnn.Net() as net:
                net.load_param(f"{export_path_base}.ncnn.param")
                net.load_model(f"{export_path_base}.ncnn.bin")
                with net.create_extractor() as ex:
                    ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())
                    _, out0 = ex.extract("out0")
                    depth_ncnn_raw = torch.from_numpy(np.array(out0)).unsqueeze(0)
                    # 应用与 PyTorch 版本相同的插值后处理
                    depth_ncnn = torch.nn.functional.interpolate(depth_ncnn_raw[:, None], (h, w), mode="bilinear",
                                                                 align_corners=True)[0, 0]
                    depth_ncnn = depth_ncnn.numpy()

            # 使用 matplotlib 可视化结果
            print("\n正在可视化结果...")
            plt.figure(figsize=(14, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(depth_pytorch, cmap='inferno')
            plt.title('PyTorch Depth Map')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(depth_ncnn, cmap='inferno')
            plt.title('NCNN Depth Map')
            plt.axis('off')

            plt.suptitle(f'DepthAnythingV2 ({encoder_type}) Comparison', fontsize=16)
            plt.show()

        except Exception as e:
            print(f"在图片处理过程中发生错误: {e}")

    print("\n导出过程成功完成!")


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