# 缪子 g-2 实验 Toy MC 模拟与数据拟合 (Muon g-2 Toy MC Simulation)

本项目包含一套用于模拟缪子反常磁矩（Muon $g-2$）实验的 Toy Monte Carlo (MC) 脚本。该程序组模拟了极化缪子在储藏环中的衰变过程，生成正电子的运动学数据，通过能量阈值筛选构建 Wiggle Plot（高能正电子计数随时间的变化），并对模拟数据进行物理参数拟合以提取反常自旋进动角频率 $\omega_a$。

## 📦 依赖库 (Dependencies)

在运行本程序之前，请确保您的 Python 环境中安装了以下依赖库：

```bash
pip install numpy matplotlib pandas scipy
```

## 📂 项目结构与文件说明

```
2026-Summer-Research-Internship
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
│
├── analysis.py
├── constants.py
├── generation.py
│
└──plot/
   ├── wiggle_line_fit_plot.png
   ├── wiggle_line_fit_plot_with_fit.png
   ├── wiggle_plot_time_mod_100.png
   └── wiggle_plot_time_mod_100_with_fit.png
```


本项目由以下三个核心 Python 脚本组成，它们构成了一个完整的数据生成、处理到拟合的工作流：

* **`constants.py`**
    * **功能**：集中存储模拟和拟合所需的物理常数与超参数。
    * **内容**：包含缪子/电子质量、洛伦兹因子 ($\gamma$)、实验室系下的寿命 ($\tau_{lab}$)、回旋频率 ($\omega_c$)、反常自旋进动角频率 ($\omega_a$)、模拟事件数 ($N = 5 \times 10^7$) 以及能量阈值等。

* **`generation.py`**
    * **功能**：核心 Toy MC 模拟器。
    * **内容**：根据指数分布生成缪子衰变时间，并通过舍选法 (Rejection Sampling) 生成正电子的运动学参数，随后进行洛伦兹变换将其转换至实验室坐标系。
    * **输出**：将探测器截获的时间和能量数据保存为 `Data.csv`。

* **`analysis.py`**
  * **功能**：数据预处理、拟合与可视化。
  * **内容**：读取生成的模拟数据，应用能量阈值筛选高能正电子（由阈值 $E > 1.7 \text{ GeV}$ 设定）。将筛选后的事件按时间装箱，并根据五参数公式进行曲线拟合，提取反常自旋进动频率 $\omega_a$。
  * **输出**：直方图数据导出为 `Counts.csv` 供检查；拟合结果保存为 `fit_results.csv`；拟合参数和误差也会在控制台输出；散点图与拟合结果图保存至 `plot/` 目录下（包括 `wiggle_line_fit_plot.png`、`wiggle_line_fit_plot_with_fit.png` 等，以及模 100 微秒的折叠图）。

## 🚀 运行指南 (Workflow)

一行命令完成整个模拟与分析流程：

```bash
python generation.py && python analysis.py
```


**第一步：生成模拟数据**
这将会运行一段时间（取决于您的 CPU），因为程序将生成 5000 万个事件的运动学数据。
```bash
python generation.py
```
*预期输出*：生成 `Data.csv` 文件。（注：请确保 `analysis.py` 中读取的文件名配置与此一致）

**第二步：筛选数据与绘图**
从生成的全量数据中提取高于能量阈值的高能正电子，生成时间分布直方图并进行曲线拟合。
```bash
python analysis.py
```
*预期输出*：生成 `Counts.csv` 和 `fit_results.csv` 并保存相关可视化图表到 `plot/` 目录下.


