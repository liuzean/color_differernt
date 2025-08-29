#!/usr/bin/env python

"""
色差分析应用启动器
"""


#import trace_fs
import argparse
import os
import socket
import sys

# 配置Matplotlib使用非交互式后端，避免线程警告
import matplotlib as plt

from interface.gui import create_interface

plt.use("Agg")  # 必须在导入pyplot之前设置
# 设置全局字体为黑体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 应用信息
APP_NAME = "色差分析系统"
APP_VERSION = "2.0.0"
APP_AUTHOR = "GDUT"

# 创建Gradio界面 - 确保变量名为demo，这样Gradio CLI可以找到它
demo = create_interface()


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_free_port(start_port=7860, max_attempts=100):
    """查找可用端口"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(
        f"无法在{start_port}-{start_port + max_attempts - 1}范围内找到可用端口"
    )


def print_welcome_message():
    """打印欢迎信息"""
    print("\033[1;36m")  # 设置颜色为青色
    print("=" * 60)
    print(f"  {APP_NAME} v{APP_VERSION}")
    print("=" * 60)
    print(f"  作者: {APP_AUTHOR}")
    print("  功能: 图像对齐与色差分析")
    print("\033[0m")  # 恢复默认颜色


def main():
    # 打印欢迎信息
    print_welcome_message()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--port", type=int, default=7860, help="Web服务器端口号")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web服务器地址")
    parser.add_argument("--share", action="store_true", help="是否创建公共分享链接")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument("--auto-port", action="store_true", help="自动查找可用端口")
    args = parser.parse_args()

    # 确保配置文件存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("将创建默认配置...")
        try:
            # 从界面模块导入默认配置
            from interface.gui import load_config, save_config

            config = load_config()
            save_config(config, args.config)
            print(f"已创建默认配置文件：{args.config}")
        except Exception as e:
            print(f"创建配置文件时出错: {str(e)}")
            return 1

    # 检查端口
    port = args.port
    if args.auto_port or is_port_in_use(port):
        try:
            port = find_free_port(start_port=port)
            print(f"端口 {args.port} 已被占用，自动切换到 {port}")
        except Exception as e:
            print(f"查找可用端口时出错: {str(e)}")
            return 1

    # 创建并启动界面
    print("=" * 50)
    print("色差分析应用启动中...")
    print(f"配置文件: {args.config}")
    print(f"服务地址: http://{args.host}:{port}")
    if args.share:
        print("公开分享模式已启用，将生成一个公共可访问链接")
    print("=" * 50)

    try:
        demo.launch(server_name=args.host, server_port=port, share=args.share)
        return 0
    except Exception as e:
        print(f"启动应用时出错: {str(e)}")
        print("提示：如果端口被占用，请尝试使用 --auto-port 参数自动选择端口")
        return 1


if __name__ == "__main__":
    sys.exit(main())
