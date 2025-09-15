import os

import numpy as np
from PIL import Image, ImageCms


def convert_color_space_array(
    input_array: np.ndarray,
    output_image_path: str | None = None,
    srgb: str | None = None,
    cmyk: str | None = None,
    to_cmyk: bool = True,
) -> "tuple[np.ndarray, Image.Image]":
    """
    颜色空间转换函数，支持sRGB与CMYK之间的转换，输入为numpy数组

    Args:
        input_array (np.ndarray): 输入图像数组
            - 如果to_cmyk=True：3通道BGR格式数组 (来自cv2.imread)
            - 如果to_cmyk=False：4通道CMYK格式数组
        output_image_path (str, optional): 输出图像路径。如果未指定，不保存文件
        srgb (str, optional): sRGB ICC文件路径。如果未指定，使用项目默认的sRGB IEC61966-21.icc
        cmyk (str, optional): CMYK ICC文件路径。如果未指定，使用项目默认的JapanColor2001Coated.icc
        to_cmyk (bool): True转换为CMYK，False转换为sRGB

    Returns:
        tuple[np.ndarray, PIL.Image]: (转换后的numpy数组, PIL图像对象)
    """
    # 获取项目根目录下的ICC文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icc_dir = os.path.join(script_dir, "icc")

    # 设置默认ICC文件路径
    if srgb is None:
        srgb_profile_path = os.path.join(icc_dir, "sRGB IEC61966-21.icc")
    else:
        srgb_profile_path = srgb

    if cmyk is None:
        cmyk_profile_path = os.path.join(icc_dir, "JapanColor2001Coated.icc")
    else:
        cmyk_profile_path = cmyk

    # 检查ICC文件是否存在
    if not os.path.exists(srgb_profile_path):
        raise FileNotFoundError(f"sRGB ICC profile not found: {srgb_profile_path}")

    if not os.path.exists(cmyk_profile_path):
        raise FileNotFoundError(f"CMYK ICC profile not found: {cmyk_profile_path}")

    # Only log the first time profiles are loaded per session
    if not hasattr(convert_color_space_array, "_profiles_logged"):
        print("ICC Color Conversion System Ready: sRGB ↔ CMYK")
        convert_color_space_array._profiles_logged = True

    # 检查输入数组
    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a numpy array")

    if len(input_array.shape) != 3:
        raise ValueError("Input array must be a 3-dimensional array")

    try:
        # 根据输入类型和转换方向处理输入数组
        if to_cmyk:
            # sRGB to CMYK：期望3通道BGR输入
            if input_array.shape[2] != 3:
                raise ValueError(
                    "For sRGB to CMYK conversion, input array must be 3-channel (BGR)"
                )
            # 将BGR转换为RGB (如果输入是BGR格式的数组)
            rgb_array = input_array[..., ::-1]  # Simple BGR to RGB channel swap
            img = Image.fromarray(rgb_array)
        else:
            # CMYK to sRGB：期望4通道CMYK输入
            if input_array.shape[2] != 4:
                raise ValueError(
                    "For CMYK to sRGB conversion, input array must be 4-channel (CMYK)"
                )
            # 直接使用CMYK数组创建PIL图像
            img = Image.fromarray(input_array, mode="CMYK")

        if to_cmyk:
            # sRGB to CMYK
            input_profile = ImageCms.ImageCmsProfile(srgb_profile_path)
            output_profile = ImageCms.ImageCmsProfile(cmyk_profile_path)
            output_mode = "CMYK"
        else:
            # CMYK to sRGB
            input_profile = ImageCms.ImageCmsProfile(cmyk_profile_path)
            output_profile = ImageCms.ImageCmsProfile(srgb_profile_path)
            output_mode = "RGB"

        # 转换图像
        converted_image = ImageCms.profileToProfile(
            img,
            inputProfile=input_profile,
            outputProfile=output_profile,
            renderingIntent=ImageCms.Intent.PERCEPTUAL,
            outputMode=output_mode,
        )

        # 确保转换成功
        if converted_image is None:
            raise RuntimeError("Color conversion failed")

        # 将PIL图像转换为numpy数组
        converted_array = np.array(converted_image)

        # 保存转换后的图像（如果指定了输出路径）
        if output_image_path:
            # 获取输出格式
            output_ext = os.path.splitext(output_image_path)[1].lower()

            # CMYK模式不支持PNG格式，转换为TIFF
            if to_cmyk and output_ext in [".png", ".gif", ".bmp"]:
                base_name = os.path.splitext(output_image_path)[0]
                output_image_path = f"{base_name}.tiff"
                output_ext = ".tiff"
                print(
                    f"Note: Format changed to TIFF (CMYK mode doesn't support {os.path.splitext(output_image_path)[1]})"
                )

            # 根据格式调整保存参数
            save_kwargs = {}
            if output_ext in [".jpg", ".jpeg"]:
                save_kwargs = {"quality": 95, "optimize": True}
            elif output_ext in [".tiff", ".tif"]:
                save_kwargs = {"compression": "lzw"}
            elif output_ext == ".png":
                save_kwargs = {"optimize": True}

            converted_image.save(output_image_path, **save_kwargs)
            print(f"Successfully converted and saved to {output_image_path}")

        return converted_array, converted_image

    except Exception as e:
        print(f"Error converting image: {e}")
        raise


def get_available_icc_profiles() -> dict:
    """
    获取项目中可用的ICC配置文件信息

    Returns:
        dict: 包含可用ICC文件路径的字典
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icc_dir = os.path.join(script_dir, "icc")

    profiles = {}

    if os.path.exists(icc_dir):
        for filename in os.listdir(icc_dir):
            if filename.endswith(".icc") or filename.endswith(".icm"):
                filepath = os.path.join(icc_dir, filename)
                profiles[filename] = filepath

    return profiles


# 便捷函数
def srgb_to_cmyk_array(
    input_array: np.ndarray,
    output_image_path: str | None = None,
    srgb: str | None = None,
    cmyk: str | None = None,
) -> "tuple[np.ndarray, Image.Image]":
    """
    将sRGB图像数组转换为CMYK格式

    Args:
        input_array (np.ndarray): 输入图像数组 (BGR格式)
        output_image_path (str, optional): 输出图像路径
        srgb (str, optional): sRGB ICC文件路径
        cmyk (str, optional): CMYK ICC文件路径

    Returns:
        tuple[np.ndarray, PIL.Image]: (转换后的numpy数组, PIL图像对象)
    """
    return convert_color_space_array(
        input_array, output_image_path, srgb, cmyk, to_cmyk=True
    )


def cmyk_to_srgb_array(
    input_array: np.ndarray,
    output_image_path: str | None = None,
    srgb: str | None = None,
    cmyk: str | None = None,
) -> "tuple[np.ndarray, Image.Image]":
    """
    将CMYK图像数组转换为sRGB格式

    Args:
        input_array (np.ndarray): 输入图像数组 (CMYK格式，4通道)
        output_image_path (str, optional): 输出图像路径
        srgb (str, optional): sRGB ICC文件路径
        cmyk (str, optional): CMYK ICC文件路径

    Returns:
        tuple[np.ndarray, PIL.Image]: (转换后的numpy数组, PIL图像对象)
    """
    return convert_color_space_array(
        input_array, output_image_path, srgb, cmyk, to_cmyk=False
    )
