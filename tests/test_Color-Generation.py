import matplotlib.pyplot as plt

def cmyk_to_rgb(c, m, y, k):
    """CMYK 转 RGB"""
    r = 255 * (1 - c/100) * (1 - k/100)
    g = 255 * (1 - m/100) * (1 - k/100)
    b = 255 * (1 - y/100) * (1 - k/100)
    return (int(r), int(g), int(b))

def show_colors(Detected, Standard):
    """
    显示 Detected 和 Standard 颜色
    - 如果是 RGB (3个值)，直接显示
    - 如果是 CMYK (4个值)，转换为 RGB 显示
    """
    def process_color(color):
        if len(color) == 3:  # RGB
            return color, f"RGB {color}"
        elif len(color) == 4:  # CMYK
            rgb = cmyk_to_rgb(*color)
            return rgb, f"CMYK {color} → RGB {rgb}"
        else:
            raise ValueError("输入必须是3个(RGB)或4个(CMYK)分量")

    detected_rgb, detected_label = process_color(Detected)
    standard_rgb, standard_label = process_color(Standard)

    # 归一化到 [0,1]
    detected_rgb_norm = tuple([v/255 for v in detected_rgb])
    standard_rgb_norm = tuple([v/255 for v in standard_rgb])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow([[detected_rgb_norm]])
    axes[0].set_title(f"Detected\n{detected_label}")
    axes[0].axis("off")

    axes[1].imshow([[standard_rgb_norm]])
    axes[1].set_title(f"Standard\n{standard_label}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例
    Detected = (183, 218, 231)     # CMYK
    Standard = (221, 198, 215)     # RGB
    show_colors(Detected, Standard)