# 缪子 g-2 Toy MC 与实验数据比较

本项目包含一套用于模拟缪子反常磁矩（Muon $g-2$）实验的 Toy Monte Carlo (MC) 脚本。该程序组模拟了极化缪子在储藏环中的衰变过程，生成正电子的运动学数据，通过能量阈值筛选构建 Wiggle Plot（高能正电子计数随时间的变化），且对模拟数据进行物理参数拟合以提取反常自旋进动角频率 $\omega_a$，并将 Toy 结果与共享的真实 ROOT 数据进行比较。更完整的方法、图像和讨论请见技术报告：`tex/main.pdf`。

## 一行命令运行


```bash
pip install -r requirements.txt
python main.py
```

## 仓库结构

```text
2026-Summer-Research-Internship/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── main.py
├── constants.py
├── generation.py
├── analysis.py
├── real_analysis.py
├── together.py
│
├── Data.csv
├── Counts.csv
├── fit_results.csv
├── real_fit_results.csv
├── residuals.csv
├── real_residuals.csv
├── run2.root
│
├── plot/
│   ├── FAKE*.png
│   ├── REAL*.png
│   ├── COMP_RES.png
│   ├── COMP_FFT.png
│   └── ...
│
└── tex/
    ├── main.tex
    ├── ref.bib
    ├── main.pdf
    └── ...
```



## 各脚本功能

* `main.py`
  * 作为单命令运行的入口，依次调用其他脚本完成整个流程。

* `constants.py`
  * 统一存放项目中使用的物理常数与分析参数。

* `generation.py`
  * 用于生成 Toy MC 事件数据。主要内容包括：
    1. 按指数分布生成缪子衰变时间；
    2. 根据 Michel spectrum 与极化方向生成正电子运动学变量；
    3. 进行实验室系变换，得到时间、能量、位置和动量信息；
    4. 将事件级数据保存为表格文件。
  * 输出： `Data.csv`

* `analysis.py`
  * 用于分析 Toy MC 数据。主要流程包括：
    1. 读取 Toy 数据文件；
    2. 根据能量阈值筛选高能正电子；
    3. 构造时间谱直方图；
    4. 使用五参数 wiggle 模型进行拟合；
    5. 计算并保存 residual；
    6. 对原始信号和 residual 做 FFT；
    7. 生成对应图像。
  * 输出：`Counts.csv`，`fit_results.csv`，`residuals.csv`，`plot/FAKE*.png`，`plot/FAKE_FIT*.png`，`plot/FAKE_WIGGLE*.png`，`plot/FAKE_WIGGLE_FIT*.png`，`plot/FAKE_RES*.png`，`plot/FAKE_FFT*.png`


* `real_analysis.py`
  * 用于分析真实 ROOT 数据。和 `analysis.py` 类似，但针对 ROOT 文件格式进行数据读取和处理。


* `together.py`
  * 用于比较 Toy 数据和真实数据的 residual 与频谱。主要流程包括：
    1. 读取 `residuals.csv`
    2. 读取 `real_residuals.csv`
    3. 绘制两组 residual 的对比图
    4. 分别计算两组 residual 的 FFT
    5. 绘制频谱对比图
  * 输出：`plot/COMP_RES.png`，`plot/COMP_FFT.png`



## 输出内容

典型输出包括：

* 拟合结果：`fit_results.csv`、`real_fit_results.csv`
* residual 文件：`residuals.csv`、`real_residuals.csv`
* `plot/` 目录下的图像
* `tex/` 目录下的报告源码及 PDF


