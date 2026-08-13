# TA-STGCN: A Hybrid Spatial-Temporal Graph Neural Network for Traffic Flow Forecasting

This repository contains the official implementation of a comprehensive research framework for evaluating Spatial-Temporal Graph Neural Networks (STGNNs) on Traffic Flow Forecasting. We propose **TA-STGCN**, a hybrid architecture integrating Temporal Convolutional Networks (TCN), Graph Convolution (GCN), and Temporal Self-Attention.

The project features a rigorous **4-Stage Experimental Pipeline** explicitly designed for high-impact IEEE journal submissions. It encompasses robust 5-seed statistical benchmarks, Computer Vision perception baseline testing, Noise Robustness studies, and Explainable AI (XAI) visualizations (Grad-CAM++ and Temporal Attention heatmaps).

---

## 🛠️ Experimental Pipeline Overview

The codebase is fully orchestrated by a master bash script (`run_all_experiments.sh`) which executes 4 consecutive evaluation stages:

### [1/4] Stage 2: Graph Forecasting Benchmark (`benchmark_5seeds.py`)
- Rigorous 5-seed statistical evaluation of our proposed model against 6 state-of-the-art baselines:
  - **STGCN** (IJCAI 2018)
  - **Graph WaveNet** (IJCAI 2019)
  - **ASTGCN** (AAAI 2019)
  - **DSTAGNN** (ICML 2022)
  - **MegaCRN** (AAAI 2023)
  - **STAEformer** (CIKM 2023)
- Models are scaled properly for fair parameter-count comparisons.
- Automatically generates multi-horizon Mean Absolute Error (MAE / MAPE) curves, complexity tables (Params/Memory), and 95% Confidence Intervals (CI).

### [2/4] Temporal Attention Interpretability (`train_and_visualize_attention.py`)
- Explains the decision-making process of **TA-STGCN**.
- Dynamically selects representative peak and off-peak traffic snapshots from the test dataset based on unnormalized traffic volume.
- Generates **Temporal Self-Attention Heatmaps** comparing the model's receptive focus across different traffic congestion scenarios (e.g., Night Off-Peak vs. Morning Peak).

### [3/4] Stage 1: Vision Perception Benchmark (`train_counting.py`)
- Evaluates cutting-edge Computer Vision models (ResNet, EfficientNet, ViT, ConvNeXt, MobileNet) to estimate real-world vehicle volume (Cars & Bikes) from traffic camera feeds.
- Computes **Grad-CAM++** to visually explain the internal feature attribution and spatial focus of the Vision models.
- Exports real-world statistical vision noise profiles.

### [4/4] Stage 3: Noise Robustness Study (`run_noise_robustness_study.py`)
- Automatically reads the empirical noise statistics (MAE) derived from the Vision Perception phase.
- Injects progressive, scientifically-calibrated Gaussian noise into the forecasting input.
- Analyzes and plots STGNN model resilience against imperfect computer vision sensory inputs.

---

## 🚀 How to Run

### Requirements
```bash
pip install torch numpy pandas matplotlib seaborn tqdm openpyxl tabulate scikit-learn opencv-python
```

### Data Preparation
Ensure the following files are present in your configured data directory:
1. `Graph_fix_py_3.xlsx` (Spatial Adjacency Matrix)
2. `count_7_7_merg_sort_fix_fill.csv` (Time-series traffic volume data containing `Timestamp`, `STT` (Node ID), `Car Count`, and `Bike Count`).

### Execution
Simply run the master shell script to execute the entire 4-stage pipeline sequentially. Before running each stage, the script will automatically `git pull` the latest changes.
```bash
bash run_all_experiments.sh
```

*(Alternatively, you can run individual scripts using `python <script_name.py>`)*

---

## 📊 Experimental Setup
- **Dataset Split**: Training / Validation / Testing split ordered chronologically.
- **Input Horizon (`T_in`)**: 120 minutes of historical data (24 time steps at 5 mins/step).
- **Output Horizon (`Horizon`)**: Forecasting up to 30 minutes into the future (6 consecutive time steps).
- **Vehicle Channels**: Parallel prediction of two distinct vehicle classes (Cars and Bikes).

---

## 📝 Generated Artifacts
All experiments automatically generate professional, publication-ready artifacts:
- `paper/fig/`: Contains high-resolution `.pdf` and `.png` plots for academic inclusion.
- `*_report.md`: Markdown tables reporting rigorous statistical metrics, ablation studies, and computational profiling.
- `model/`: Directory storing the best weights/checkpoints for all trained models.
