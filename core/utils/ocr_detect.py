import base64
import json
import os
from typing import Any

import requests


def ocr_image(
    image_path: str, host: str = "localhost", port: int = 1224
) -> dict[str, Any]:
    """对图像文件执行OCR识别

    参数:
        image_path: 图像文件路径
        host: Umi-OCR HTTP服务器主机
        port: Umi-OCR HTTP服务器端口

    返回:
        OCR识别结果字典
    """
    # 检查文件是否存在
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"未找到图像文件: {image_path}")

    # 构造API URL
    base_url = f"http://{host}:{port}/api"

    # 读取图像文件为二进制
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 将图像编码为base64
    base64_data = base64.b64encode(image_data).decode("utf-8")

    # 准备请求数据
    payload = {
        "base64": base64_data,
        "options": {
            "data.format": "text",
        },
    }

    # 发送OCR请求
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        f"{base_url}/ocr", data=json.dumps(payload), headers=headers
    )
    response.raise_for_status()

    return json.loads(response.text)


def get_ocr_options(host: str = "localhost", port: int = 1224) -> dict[str, Any]:
    """获取当前OCR选项设置

    参数:
        host: Umi-OCR HTTP服务器主机
        port: Umi-OCR HTTP服务器端口

    返回:
        OCR选项设置字典
    """
    base_url = f"http://{host}:{port}/api"
    response = requests.get(f"{base_url}/ocr/get_options")
    return response.json()


def set_ocr_options(
    options: dict[str, Any], host: str = "localhost", port: int = 1224
) -> dict[str, Any]:
    """设置OCR选项

    参数:
        options: 要设置的选项字典
        host: Umi-OCR HTTP服务器主机
        port: Umi-OCR HTTP服务器端口

    返回:
        操作结果字典
    """
    base_url = f"http://{host}:{port}/api"
    response = requests.post(f"{base_url}/ocr/set_options", json=options)
    return response.json()


# def main():
#     """主函数，处理命令行参数并执行OCR"""
#     parser = argparse.ArgumentParser(description="使用Umi-OCR从图像中提取文本")
#     parser.add_argument("image", help="图像文件路径")
#     parser.add_argument("--host", default="127.0.0.1", help="OCR服务器主机")
#     parser.add_argument("--port", type=int, default=1224, help="OCR服务器端口")

#     args = parser.parse_args()

#     try:
#         # 执行OCR识别
#         result = ocr_image(args.image, args.host, args.port)

#         # 打印识别结果
#         if "data" in result:
#             print("OCR识别结果:")
#             print(result["data"])
#             # 按空格分割文本并显示单词列表
#             words = result["data"].split()
#             print(f"\n单词列表({len(words)}个):")
#             for i, word in enumerate(words, 1):
#                 print(f"{i}. {word}")
#         else:
#             print(f"OCR识别失败: {result}")

#     except Exception as e:
#         print(f"错误: {str(e)}")
#         return 1

#     return 0


# if __name__ == "__main__":
#     main()
