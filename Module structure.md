"""
模块功能依赖图
"""


color_checker.py(第三级) 依赖关系:
│
├── CMYKConfigManager (核心配置管理)
│   ├── 预设管理 → get_preset_names(), load_preset()
│   ├── 配置验证 → load_from_yaml()
│   ├── 数据获取 → get_palette_data(), get_layout_config()
│   └── 序列化 → get_yaml_string()
│
└── ColorPaletteGenerator (核心生成引擎)
    └── 文件生成 → generate_palette()
        ├── 预览生成 (PDF, 150 DPI)
        ├── 个人渐变 (PDF + TIFF, 300 DPI)
        └── 步骤文件 (PDF + TIFF, 300 DPI)



colorbar_analysis.py（第三级） 功能依赖:
│
├── 核心分析模块 (core/block_detection/)
│   └── pure_colorbar_analysis_for_gradio()
│       ├── YOLO检测参数处理
│       ├── 色块检测与分析
│       ├── 纯色分析算法
│       └── 图像标注生成
│
├── 共享结果组件 (interface/components/)
│   └── update_shared_results_display()
│       ├── 颜色条数据格式化
│       ├── HTML结果生成
│       └── 可视化展示
│
└── 数据流向:
    输入图像 → 纯色分析 → 结果格式化 → HTML显示
         ↓           ↓           ↓          ↓
    PIL.Image → colorbar_data → results_html → Gradio界面



ground_truth_colorbar_demo.py（第三级） 功能依赖:
│
├── 调色板生成模块 (core/color/)
│   ├── ColorPaletteGenerator
│   │   ├── generate_palette() → 生成PDF/TIFF调色板文件
│   │   └── convert_cmyk_tiff_to_png() → 格式转换用于显示
│   │
│   ├── ground_truth_checker
│   │   ├── generate_reference_chart() → 生成标准参考图表
│   │   └── get_palette_yaml() → 获取标准配置YAML
│   │
│   └── cmyk_to_rgb() → CMYK颜色空间转换
│
├── 检测分析模块 (core/block_detection/)
│   └── pure_colorbar_analysis_for_gradio() → 纯色分析算法
│
├── 共享组件 (interface/components/)
│   └── update_shared_results_display() → 结果格式化显示
│
└── 数据流向:
    配置选择 → 图像生成 → 颜色分析 → 结果比较 → HTML显示
         ↓         ↓         ↓         ↓         ↓
    demo_configs → PIL.Image → colorbar_data → gt_results → results_html



preview.py（第三级） 功能依赖:
│
├── ICC颜色转换模块 (core/color/icc_trans.py)
│   ├── srgb_to_cmyk_array()
│   │   ├── 接收sRGB BGR数组
│   │   └── 输出CMYK数组和元数据
│   │
│   └── cmyk_to_srgb_array()
│       ├── 接收CMYK数组
│       └── 输出RGB数组和PIL图像
│
├── 外部依赖 (标准库/第三方)
│   ├── PIL.Image → 图像处理
│   ├── cv2 → OpenCV颜色空间转换
│   ├── numpy → 数组操作
│   └── gradio → UI组件
│
└── 数据流向:
    输入图像 → 格式转换 → 颜色空间变换 → 预览输出
         ↓          ↓           ↓           ↓
    PIL.Image → numpy数组 → ICC转换 → PIL.Image



results.py（第三级） 功能依赖:
│
├── 外部依赖:
│   └── gradio → Web UI框架
│       ├── gr.Tabs() → 标签页容器
│       ├── gr.TabItem() → 标签页项
│       ├── gr.Row() → 行布局
│       └── gr.Image() → 图像显示组件
│
├── UI组件结构:
│   ├── Tab 1: "Alignment Result"
│   │   ├── aligned_image (对齐结果)
│   │   └── diff_map (差异映射)
│   │
│   └── Tab 2: "Color Difference"
│       ├── Row 1: heatmap + heatmap_colorbar
│       ├── Row 2: overlayed_heatmap + highlighted
│       └── Row 3: block_heatmap + overlay_blocks
│
└── 数据流向:
    无内部数据处理 → 纯UI布局定义 → 返回组件引用
         ↓              ↓              ↓
    静态定义 → Gradio组件创建 → 组件实例返回


settings.py（第三级） 功能依赖:
│
├── ICC配置管理模块 (core/color/icc_trans.py)
│   └── get_available_icc_profiles()
│       ├── 扫描系统ICC配置文件
│       ├── 返回可用的颜色配置文件字典
│       └── 用于构建颜色空间选择列表
│
├── 外部依赖 (标准库/第三方)
│   └── gradio → UI组件框架
│       ├── gr.Tabs() → 标签页容器
│       ├── gr.TabItem() → 标签页项
│       ├── gr.Row() / gr.Column() → 布局组件
│       ├── gr.Markdown() → 文档显示
│       ├── gr.Dropdown() → 下拉选择框
│       └── gr.Number() → 数值输入框
│
└── 数据流向:
    系统ICC文件 → get_available_icc_profiles() → profile_names → Dropdown选项
         ↓                    ↓                        ↓            ↓
    文件系统扫描 → 配置文件字典 → 选项列表构建 → UI组件显示