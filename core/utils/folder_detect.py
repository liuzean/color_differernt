import os


def find_images_in_folder(
    base_path: str, folder_name: str, file_extensions: list[str] = None
) -> list[str]:
    """在基础路径的第一层目录中查找指定文件夹并返回其中的图像文件

    参数:
        base_path: 基础路径(数据库路径)
        folder_name: 要查找的文件夹名
        file_extensions: 要查找的文件扩展名列表

    返回:
        图像文件路径列表，如果文件夹不存在则为空列表
    """
    # 如果没有提供扩展名，使用默认的图像扩展名
    if file_extensions is None:
        file_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".gif",
            ".webp",
        ]

    # 构造完整的文件夹路径（只检查第一层目录）
    folder_path = os.path.join(base_path, folder_name)

    # 如果文件夹不存在，返回空列表
    if not os.path.isdir(folder_path):
        print(f"未找到文件夹: {folder_path}")
        return []

    # 查找所有图像文件
    image_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            # 检查文件扩展名是否在指定列表中
            _, ext = os.path.splitext(file_path.lower())
            if ext in file_extensions:
                image_files.append(file_path)

    return image_files


# def main():
#     """主函数，处理命令行参数并查找图像"""
#     parser = argparse.ArgumentParser(description="在指定文件夹中查找图像文件")
#     parser.add_argument("base_path", help="基础路径")
#     parser.add_argument("folder_name", help="要查找的文件夹名称")
#     parser.add_argument(
#         "--ext", nargs="+", default=None, help="要查找的文件扩展名（例如 jpg png）"
#     )

#     args = parser.parse_args()

#     # 格式化扩展名
#     extensions = None
#     if args.ext:
#         extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.ext]

#     try:
#         # 查找图像文件
#         image_files = find_images_in_folder(args.base_path, args.folder_name, extensions)

#         # 打印结果
#         if image_files:
#             print(f"在 {os.path.join(args.base_path, args.folder_name)} 中找到 {len(image_files)} 个图像文件:")
#             for path in image_files:
#                 print(f"  - {path}")
#         else:
#             print(f"在 {os.path.join(args.base_path, args.folder_name)} 中未找到图像文件")

#     except Exception as e:
#         print(f"错误: {str(e)}")
#         return 1

#     return 0

# if __name__ == "__main__":
#     sys.exit(main())
