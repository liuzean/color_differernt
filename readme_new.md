"""
调用关系图:
"""

app.py（第一级）
│
├── interface/
|    └───gui.py(第二级的主文件)
│          ├── create_interface()        # 始终调用
│          ├── load_config()            # 条件调用（配置文件不存在时）
│          └── save_config()            # 条件调用（配置文件不存在时）



gui.py（第二级）
│
├── components/
│   ├── color_checker.py （三级）               # create_color_checker_ui()
│   ├── colorbar_analysis.py （三级）           # create_colorbar_analysis_ui()
│   ├── ground_truth_colorbar_demo.py （三级）  # create_ground_truth_colorbar_demo_ui()
│   ├── preview.py （三级）                     # create_preview_ui(), update_preview()
│   ├── results.py   （三级）                   # create_results_ui()
│   └── settings.py  （三级）                   # create_settings_ui()
│
├── config.py                         # load_config()
│
└── handlers/
    └── callbacks.py                  # process_images_handler(), save_config_handler()



color_checker.py (第三级)
│
└── 项目内部模块导入:
    │
    ├── core/
    │   └── color/
    │       ├── palette_configs.py (第四级)
    │       │   └── CMYKConfigManager
    │       │       ├── get_preset_names()        # 获取预设名称列表
    │       │       ├── get_yaml_string()         # 获取YAML配置字符串
    │       │       ├── load_from_yaml()          # 加载并验证YAML配置
    │       │       ├── load_preset()             # 加载指定预设配置
    │       │       ├── get_palette_data()        # 获取调色板数据
    │       │       └── get_layout_config()       # 获取布局配置
    │       │
    │       └── palette_generator.py (第四级)
    │           └── ColorPaletteGenerator
    │               └── generate_palette()        # 生成调色板文件(PDF/TIFF)




colorbar_analysis.py (第三级)
│
└── 项目内部模块导入:
    │
    ├── core/
    │   └── block_detection/
    │       └── pure_colorbar_analysis.py (第四级)
    │           └── pure_colorbar_analysis_for_gradio()
    │               ├── 输入参数:
    │               │   ├── input_image (PIL.Image)
    │               │   ├── confidence_threshold
    │               │   ├── box_expansion
    │               │   ├── block_area_threshold
    │               │   ├── block_aspect_ratio
    │               │   ├── min_square_size
    │               │   └── purity_threshold
    │               │
    │               └── 返回值:
    │                   ├── annotated_image (标注后的图像)
    │                   ├── colorbar_data (颜色条数据)
    │                   ├── analysis_report (分析报告)
    │                   └── total_blocks (检测到的色块总数)
    │
    └── interface/components/
        └── shared_results.py (第三级)
            └── update_shared_results_display()
                ├── 输入: colorbar_data
                └── 返回: results_html (格式化的HTML显示)


ground_truth_colorbar_demo.py (第三级)
│
└── 项目内部模块导入:
    │
    ├── core/
    │   ├── color/
    │   │   ├── ground_truth_checker.py (第四级)
    │   │   │   └── ground_truth_checker
    │   │   │       ├── generate_reference_chart()     # 生成参考图表
    │   │   │       └── get_palette_yaml()             # 获取调色板YAML配置
    │   │   │
    │   │   ├── palette_generator.py (第四级)
    │   │   │   └── ColorPaletteGenerator
    │   │   │       ├── generate_palette()             # 生成调色板文件
    │   │   │       └── convert_cmyk_tiff_to_png()     # CMYK TIFF转PNG
    │   │   │
    │   │   └── utils.py (第四级)
    │   │       └── cmyk_to_rgb()                      # CMYK到RGB颜色转换
    │   │
    │   └── block_detection/
    │       └── pure_colorbar_analysis.py (第四级)
    │           └── pure_colorbar_analysis_for_gradio() # 纯色颜色条分析
    │
    └── interface/components/
        └── shared_results.py (第三级)
            └── update_shared_results_display()        # 共享结果显示


preview.py (第三级)
│
└── 项目内部模块导入:
    │
    └── core/
        └── color/
            └── icc_trans.py (第四级)
                ├── cmyk_to_srgb_array()          # CMYK到sRGB数组转换
                │   ├── 输入: cmyk_array
                │   └── 返回: (final_rgb_array, final_image)
                │
                └── srgb_to_cmyk_array()          # sRGB到CMYK数组转换
                    ├── 输入: bgr_array
                    └── 返回: (cmyk_array, _)




results.py (第三级)
│
└── 项目内部模块导入:
    │
    └── 无项目内部模块导入
        │
        └── 仅导入外部依赖:
            └── gradio as gr  # Web UI框架



settings.py (第三级)
│
└── 项目内部模块导入:
    │
    └── core/
        └── color/
            └── icc_trans.py (第四级)
                └── get_available_icc_profiles()
                    ├── 功能: 获取可用的ICC配置文件
                    ├── 输入: 无参数
                    └── 返回: dict[str, profile_info] 或 None
