# 缪子 g-2 实验 Toy MC 模拟与数据拟合 (Muon g-2 Toy MC Simulation)

本项目包含一套用于模拟缪子反常磁矩（Muon $g-2$）实验的 Toy Monte Carlo (MC) 脚本。该程序组模拟了极化缪子在储藏环中的衰变过程，生成正电子的运动学数据，通过能量阈值筛选构建 Wiggle Plot（高能正电子计数随时间的变化），并对模拟数据进行物理参数拟合以提取反常自旋进动角频率 $\omega_a$。

## 📦 依赖库 (Dependencies)

在运行本程序之前，请确保您的 Python 环境中安装了以下依赖库：

```bash
pip install numpy pandas matplotlib scipy uproot
```
*(注：`uproot` 在 `fit.py` 中被导入，尽管在当前代码逻辑中主要使用 `numpy.genfromtxt` 读取 CSV 数据)*

## 📂 项目结构与文件说明

本项目由以下四个核心 Python 脚本组成，它们构成了一个完整的数据生成、处理到拟合的工作流：

* **`constants.py`**
    * **功能**：集中存储模拟和拟合所需的物理常数与超参数。
    * **内容**：包含缪子/电子质量、洛伦兹因子 ($\gamma$)、实验室系下的寿命 ($\tau_{lab}$)、回旋频率 ($\omega_c$)、反常自旋进动角频率 ($\omega_a$)、模拟事件数 ($N = 5 \times 10^6$) 以及能量阈值等。

* **`generate_data.py`**
    * **功能**：核心 Toy MC 模拟器。
    * **内容**：根据指数分布生成缪子衰变时间，并通过**舍选法 (Rejection Sampling)** 根据弱相互作用宇称不守恒的概率密度函数 (PDF) 生成正电子的运动学参数。随后通过洛伦兹变换将其从缪子静止系 boost 并旋转至实验室坐标系。
    * **输出**：将探测器截获的四动量、时间和位置数据保存为 `simulated_detector_data.csv`。

* **`plot.py`**
    * **功能**：数据预处理与可视化。
    * **内容**：读取生成的模拟数据，应用能量阈值筛选高能正电子（由阈值 $E > 1.7 \text{ GeV}$ 设定）。将筛选后的事件按时间装箱 (binning) 生成直方图数据。
    * **输出**：绘制散点图并保存为 `plot/wiggle_scatter_plot.png`；同时将直方图数据导出为 `wiggle_plot_data.csv` 供拟合使用。

* **`fit.py`**
    * **功能**：物理参数提取。
    * **内容**：读取 `wiggle_plot_data.csv` 中的直方图数据，使用 `scipy.optimize.curve_fit` 对数据进行标准 Wiggle 函数拟合。
    * **输出**：在终端打印拟合参数及其误差，并绘制带拟合曲线及参数说明框的最终图表，保存至 `plot/real_wiggle_plot_run6A_fit.png`。

## 🚀 运行指南 (Workflow)

请严格按照以下顺序执行脚本，因为后一个脚本依赖前一个脚本的输出文件。

**第一步：生成模拟数据**
这将会运行一段时间（取决于您的 CPU），因为使用了舍选法生成 500 万个事件的运动学数据。
```bash
python generate_data.py
```
*预期输出*：生成 `simulated_detector_data.csv` 文件。

**第二步：筛选数据与绘图**
从生成的全量数据中提取高于能量阈值的高能正电子，生成时间分布直方图。
```bash
python plot.py
```
*预期输出*：生成 `wiggle_plot_data.csv` 并保存散点图到 `plot/` 目录下。

**第三步：执行 Wiggle 拟合**
基于五参数（代码中固定了衰变寿命，使用了四个自由参数）拟合公式提取反常自旋进动频率。
```bash
python fit.py
```
*预期输出*：在终端看到拟合结果输出，并在 `plot/` 目录下生成最终的拟合结果图。

## 🧮 核心物理公式

在 `fit.py` 中，拟合程序使用了简化的理想 Wiggle 函数来描述高能正电子计数随时间的演化规律：

$$N(t) = N_0 e^{-t/\gamma\tau} \left[ 1 + A \cos(\omega_a t - \phi_0) \right]$$

* $N_0$: 归一化常数 (Normalization constant)
* $\gamma\tau$: 实验室系下的缪子寿命 ($\tau_{lab}$)
* $A$: 不对称度参数 (Asymmetry parameter)
* $\omega_a$: 缪子反常自旋进动角频率 (Anomalous precession frequency)
* $\phi_0$: 初始相位 (Initial phase)