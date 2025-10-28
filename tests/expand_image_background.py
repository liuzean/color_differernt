import os
from PIL import Image

# 输入和输出目录
#只是一个扩大图片的代码，但没什么用处，可以忽略

INPUT_DIR = r"E:\color_project\打光测试图片\经过我后期加工过的"
OUTPUT_DIR = os.path.join(INPUT_DIR, "expend")

def expand_image(image_path: str, output_path: str):
    """
    将图片扩展为原始尺寸的两倍大小：
    - 从图片的四个边分别截取宽度为 100 像素的区域，作为扩展素材。
    - 将截取的区域重复拼接到原图的四周，直到图片的尺寸达到原始图片的两倍。
    - 四个角的空缺部分用白色填充。
    """
    with Image.open(image_path) as img:
        # 获取原始图片尺寸
        original_width, original_height = img.size

        # 计算目标尺寸
        target_width = original_width * 2
        target_height = original_height * 2

        # 截取四边各 100 行/列的像素
        edge_width = 100
        left_edge = img.crop((0, 0, edge_width, original_height))  # 左侧 100 列
        right_edge = img.crop((original_width - edge_width, 0, original_width, original_height))  # 右侧 100 列
        top_edge = img.crop((0, 0, original_width, edge_width))  # 顶部 100 行
        bottom_edge = img.crop((0, original_height - edge_width, original_width, original_height))  # 底部 100 行

        # 创建新图像，背景为白色
        expanded_img = Image.new("RGB", (target_width, target_height), color=(255, 255, 255))

        # 将原图粘贴到中心
        offset_x = edge_width
        offset_y = edge_width
        expanded_img.paste(img, (offset_x, offset_y))

        # 左右扩展
        for x in range(offset_y, offset_y + original_height, edge_width):
            # 左侧
            expanded_img.paste(left_edge, (0, x))
            # 右侧
            expanded_img.paste(right_edge, (target_width - edge_width, x))

        # 上下扩展
        for y in range(offset_x, offset_x + original_width, edge_width):
            # 顶部
            expanded_img.paste(top_edge, (y, 0))
            # 底部
            expanded_img.paste(bottom_edge, (y, target_height - edge_width))

        # 保存扩大后的图片
        expanded_img.save(output_path)

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历输入目录中的所有图片文件
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)

        # 检查是否为图片文件
        if not os.path.isfile(input_path) or not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # 输出路径
        output_path = os.path.join(OUTPUT_DIR, filename)

        # 扩大图片背景
        expand_image(input_path, output_path)
        print(f"已处理: {filename}")

    print(f"所有图片已处理，输出到目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()