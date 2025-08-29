import tempfile
import traceback
import builtins
from pathlib import Path

temp_dir = tempfile.gettempdir()
print(f"监控临时目录: {temp_dir}")

# ---- Hook pathlib.Path.mkdir ----
_old_path_mkdir = Path.mkdir

def debug_path_mkdir(self, *args, **kwargs):
    try:
        if str(self).startswith(temp_dir):
            print(f"\n📁 检测到创建目录: {self}")
            traceback.print_stack(limit=5)
    except Exception as e:
        print(f"debug_path_mkdir hook 错误: {e}")
    return _old_path_mkdir(self, *args, **kwargs)

Path.mkdir = debug_path_mkdir

# ---- Hook open ----
_old_open = builtins.open

def debug_open(file, mode="r", *args, **kwargs):
    try:
        if str(file).startswith(temp_dir) and ("w" in mode or "a" in mode):
            print(f"\n⚡ 检测到文件写入: {file} (mode={mode})")
            print("调用来源:")
            traceback.print_stack(limit=5)
    except Exception as e:
        print(f"debug_open hook 错误: {e}")
    return _old_open(file, mode, *args, **kwargs)

builtins.open = debug_open
