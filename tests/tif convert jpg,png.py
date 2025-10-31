import os
from typing import Tuple
from PIL import Image, ImageOps

# 输入与输出
INPUT_DIR = r"E:\color_project\打光测试图片\二次打光\20251025彩色贴纸检测\有膜"
OUTPUT_ROOT = r"E:\projects\color-difference-feature-start-rael-opencv\test_output"

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"{SCRIPT_NAME}_results")

VALID_EXTS = {".tif", ".tiff"}

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{base}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1

def _open_first_frame(img_path: str) -> Image.Image:
    img = Image.open(img_path)
    try:
        img.seek(0)  # 确保在首帧
    except Exception:
        pass
    return img

def _to_png_image(img: Image.Image) -> Image.Image:
    # PNG 可保留透明：优先 RGBA
    if img.mode in ("RGBA", "LA"):
        return img.convert("RGBA")
    # 调色板带透明
    if img.mode == "P":
        if "transparency" in img.info:
            return img.convert("RGBA")
        else:
            return img.convert("RGB")
    # 16位/整型等 -> 8位
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        img8 = ImageOps.autocontrast(img.convert("I")).convert("L")
        return img8.convert("RGB")
    # CMYK/其他 -> RGB
    if img.mode in ("CMYK", "YCbCr"):
        return img.convert("RGB")
    # 其余：L/RGB 直接返回为 RGB/保持
    if img.mode == "RGB":
        return img
    if img.mode == "L":
        return img.convert("RGB")
    return img.convert("RGB")

def _to_jpg_image(img: Image.Image) -> Image.Image:
    # JPG 不支持透明，若有 alpha 则铺白底
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info if hasattr(img, "info") else False):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    # 16位/整型等 -> 8位
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        img8 = ImageOps.autocontrast(img.convert("I")).convert("L")
        return img8.convert("RGB")
    # CMYK/其他 -> RGB
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def convert_one(tif_path: str) -> Tuple[str, str]:
    img = _open_first_frame(tif_path)
    base = os.path.splitext(os.path.basename(tif_path))[0]

    # 目标路径（同名冲突自动加后缀）
    out_png = _unique_path(os.path.join(OUTPUT_DIR, f"{base}.png"))
    out_jpg = _unique_path(os.path.join(OUTPUT_DIR, f"{base}.jpg"))

    # PNG
    png_img = _to_png_image(img)
    png_img.save(out_png, format="PNG", optimize=True)

    # JPG
    jpg_img = _to_jpg_image(img)
    jpg_img.save(out_jpg, format="JPEG", quality=95, optimize=True)

    try:
        img.close()
    except Exception:
        pass
    return out_png, out_jpg

def main():
    _ensure_dir(OUTPUT_DIR)
    total, ok, fail = 0, 0, 0
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 递归遍历（如不需要递归，可改为 os.listdir）
    for root, _dirs, files in os.walk(INPUT_DIR):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VALID_EXTS:
                continue
            total += 1
            src = os.path.join(root, name)
            try:
                out_png, out_jpg = convert_one(src)
                ok += 1
                print(f"[OK] {src} -> {out_png} | {out_jpg}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {src} -> {e}")

    print(f"完成：共{total}个，成功{ok}，失败{fail}。")

if __name__ == "__main__":
    main()