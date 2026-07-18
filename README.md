# So sánh các mô hình Spatial-Temporal Graph Neural Networks (STGNNs) cho Dự báo Lưu lượng Giao thông

Dự án này được xây dựng nhằm nghiên cứu khoa học (NCKH), thực nghiệm và so sánh hiệu năng của các kiến trúc mạng nơ-ron đồ thị không-thời gian (Spatial-Temporal Graph Neural Networks) phổ biến trong bài toán dự báo lưu lượng giao thông (Traffic Flow Forecasting).

Nghiên cứu tập trung so sánh các baseline kinh điển và thực hiện đánh giá sâu sắc các cơ chế mô hình hóa thời gian khác nhau (**GLU, BiLSTM, TCN, Attention**) trên cùng một bộ khung không gian đồ thị có hướng (**Diffusion Graph Convolution**).

---

## 📂 Danh sách các Kiến trúc Mô hình

Dự án hiện tại hỗ trợ **7 kiến trúc mô hình độc lập**, được tối ưu hóa hiệu năng huấn luyện bằng PyTorch thuần để dễ dàng chạy thực nghiệm:

### 1. Các Kiến trúc Baseline chính
* **GCN-LSTM** ([gcn_lstm.py](file:///g:/nckh/gcn_lstm.py)):
  * **Spatial**: Graph Convolutional Network (GCN) trích xuất đặc trưng không gian tĩnh.
  * **Temporal**: Long Short-Term Memory (LSTM) học các phụ thuộc thời gian dài hạn.
* **Graph WaveNet** ([wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py)):
  * **Spatial**: Kết hợp GCN cố định cùng cơ chế ma trận kề thích ứng (Adaptive Adjacency) tự học cấu trúc đồ thị từ dữ liệu.
  * **Temporal**: Lớp Dilated Causal Convolution (Gated TCN) mở rộng receptive field nhanh chóng.
* **STGCN** ([stgcn.py](file:///g:/nckh/stgcn.py)):
  * **Spatial**: Chebyshev Spectral Graph Convolution (ChebConv).
  * **Temporal**: Lớp Temporal Gated Convolution (GLU) qua tích chập 1D.

### 2. Các Biến thể cải tiến của DCRNN (Mô hình hóa thời gian song song)
* **DCRNN-GLU** ([dcrnn_glu.py](file:///g:/nckh/dcrnn_glu.py)) [NEW - ĐỀ XUẤT]:
  * **Spatial**: Tích chập lan truyền (Diffusion Convolution) trên đồ thị có hướng.
  * **Temporal**: Lớp **Gated CNN (GLU) 1D** thay thế cho các tế bào tuần tự DCGRU giúp song song hóa học thời gian cực nhanh.
* **DCRNN-BiLSTM** ([dcrnn_bilstm.py](file:///g:/nckh/dcrnn_bilstm.py)) [NEW - ĐỀ XUẤT]:
  * **Spatial**: Tích chập lan truyền (Diffusion Convolution) trên đồ thị có hướng.
  * **Temporal**: Lớp **Bidirectional LSTM (BiLSTM)** xử lý đặc trưng chuỗi thời gian hai chiều động.
* **DCRNN-TCN** ([dcrnn_tcn.py](file:///g:/nckh/dcrnn_tcn.py)) [NEW - ĐỀ XUẤT]:
  * **Spatial**: Tích chập lan truyền (Diffusion Convolution) trên đồ thị có hướng.
  * **Temporal**: Lớp **Stacked Dilated Causal TCN** học quan hệ thời gian dài hạn bằng giãn nở exponential.
* **DCRNN-Attention** ([dcrnn_attention.py](file:///g:/nckh/dcrnn_attention.py)) [NEW - ĐỀ XUẤT]:
  * **Spatial**: Tích chập lan truyền (Diffusion Convolution) trên đồ thị có hướng.
  * **Temporal**: Lớp **Multi-Head Temporal Self-Attention** trích xuất ngữ cảnh thời gian động toàn cục.

---

## 🛠️ Cài đặt Môi trường

Cài đặt các thư viện cần thiết trước khi chạy:
```bash
pip install torch numpy pandas matplotlib tqdm openpyxl tabulate
```

---

## 📊 Chuẩn bị Dữ liệu

Cấu hình đường dẫn dữ liệu được khai báo trong lớp `Config` ở đầu các mô hình:
1. **Excel chứa Ma trận kề**: `Graph_fix_py_3.xlsx`
2. **CSV chứa Chuỗi thời gian**: `count_7_7_merg_sort_fix_fill.csv` (yêu cầu cột `Timestamp`, `STT` (Node ID), và `Total Vehicles`).

---

## 🚀 Hướng dẫn Sử dụng

### 1. Chạy Huấn luyện Từng Mô hình Độc lập
```bash
python gcn_lstm.py
python wavenet_gcn.py
python stgcn.py
python dcrnn_glu.py
python dcrnn_bilstm.py
python dcrnn_tcn.py
python dcrnn_attention.py
```
*Mỗi file mô hình đều được trang bị early stopping với độ kiên nhẫn `PATIENCE = 20`, thanh tiến trình `tqdm.auto` và tự động lưu checkpoint đạt MAE tốt nhất trên tập Validation vào thư mục `./model/`.*

### 2. Chạy So sánh Đồng thời cả 7 Mô hình (compare_models.py)
Để thực hiện thực nghiệm so sánh tất cả các kiến trúc trên cùng một tập dữ liệu dùng chung (cùng cách phân chia Train/Val/Test), hãy sử dụng script [compare_models.py](file:///g:/nckh/compare_models.py):

* **Chế độ Đánh giá nhanh (Mặc định)**: Tải các checkpoint `.pth` tốt nhất trong thư mục `model/` của cả 7 mô hình hiện có và đánh giá trên tập Test để xuất bảng so sánh chỉ số:
  ```bash
  python compare_models.py --mode eval
  ```
* **Chế độ Huấn luyện mới**: Huấn luyện tuần tự cả 7 mô hình từ đầu, tự động lưu checkpoint vào `model/` và tổng hợp bảng so sánh:
  ```bash
  python compare_models.py --mode train --epochs 100
  ```

Sau khi chạy xong, kết quả so sánh (các chỉ số **Loss**, **MAE**, **MSE**, **RMSE** trên tập Test) sẽ được hiển thị trên console dưới dạng bảng Markdown và được tự động ghi vào tệp báo cáo `comparison_report.md`.

---

## 📝 Quy ước trong Nghiên cứu khoa học
* **Độ chia dữ liệu**: Mặc định chia theo tỉ lệ thời gian `80% Train` / `10% Val` / `10% Test`.
* **Khung thời gian dự báo**: Đầu vào `T_IN` sử dụng dữ liệu lịch sử của 120 phút trước (tương đương 24 bước với bước thời gian 5 phút), dự báo trước `HORIZON = 6` bước tiếp theo (30 phút tương lai).
* **Quản lý Checkpoints**: Các checkpoint của mô hình được tự động lưu trong thư mục `model/` tương đối tại gốc thư mục chạy để đảm bảo tính di động cao (chạy được cả trên môi trường local lẫn Google Colab).
