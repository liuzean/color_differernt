import cv2
import folder_detect
import ocr_detect


def main():
    # 1. 先执行OCR识别获取文本
    image_path = (
        "C:/Users/gdut/Desktop/C_D/new_version/color_difference/photo/img008.tif"
    )
    print(f"对图像 {image_path} 执行OCR识别...")

    try:
        # 执行OCR识别
        result = ocr_detect.ocr_image(image_path)

        # 打印识别结果
        if "data" in result:
            print("OCR识别结果:")
            print(result["data"])
            # 按空格分割文本并显示单词列表
            words = result["data"].split()
            print(f"\n识别列表({len(words)}个):")
            for i, word in enumerate(words, 1):
                print(f"{i}. {word}")

            # 2. 根据OCR识别结果中的第一个词查找对应文件夹
            if words:
                for word in words:
                    search_word = word
                    print(f"\n使用第{i}个识别词 '{search_word}' 查找对应文件夹...")

                base_path = (
                    "C:/Users/gdut/Desktop/C_D/new_version/color_difference/photo"
                )
                images = folder_detect.find_images_in_folder(base_path, search_word)

                if images:
                    print(f"在文件夹 '{search_word}' 中找到 {len(images)} 个图像文件:")
                    for img in images:
                        print(f"  - {img}")
                        cv2.imshow("image", cv2.imread(img))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    print(f"未找到与 '{search_word}' 对应的文件夹或文件夹中没有图像")
            else:
                print("OCR未识别出任何文字，无法查找对应文件夹")
        else:
            print(f"OCR识别失败: {result}")

    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
