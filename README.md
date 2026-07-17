# So sánh các mô hình Spatial-Temporal Graph Neural Networks (STGNNs) cho Dự báo Lưu lượng Giao thông

Dự án này được xây dựng nhằm nghiên cứu khoa học (NCKH), thực nghiệm và so sánh hiệu năng của các kiến trúc mạng nơ-ron đồ thị không-thời gian (Spatial-Temporal Graph Neural Networks) phổ biến trong bài toán dự báo lưu lượng giao thông (Traffic Flow Forecasting).

---

## 📂 Danh sách các Kiến trúc Mô hình

Dự án hiện tại hỗ trợ **9 kiến trúc mô hình độc lập**, được tối ưu hóa hiệu năng huấn luyện bằng PyTorch thuần để dễ dàng chạy thực nghiệm:

1. **GCN-LSTM** ([gcn_lstm.py](file:///g:/nckh/gcn_lstm.py)):
   * **Spatial**: Graph Convolutional Network (GCN) trích xuất đặc trưng không gian tĩnh.
   * **Temporal**: Long Short-Term Memory (LSTM) học các phụ thuộc thời gian dài hạn.
2. **Graph WaveNet** ([wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py)):
   * **Spatial**: Kết hợp GCN cố định cùng cơ chế ma trận kề thích ứng (Adaptive Adjacency) tự học cấu trúc đồ thị từ dữ liệu.
   * **Temporal**: Lớp Dilated Causal Convolution (Gated TCN) mở rộng receptive field nhanh chóng.
3. **GAT-TCN** ([gat_tcn.py](file:///g:/nckh/gat_tcn.py)):
   * **Spatial**: Graph Attention Network (GAT) tự động học trọng số động giữa các nút.
   * **Temporal**: Stacked Temporal Convolutional Network (TCN) với dilation tăng dần.
4. **ASTGCN** ([astgcn.py](file:///g:/nckh/astgcn.py)):
   * **Spatial & Temporal Attention**: Cơ chế Attention động cả về mặt không gian (giữa các nút) và thời gian.
   * **Chebyshev GCN**: Chebyshev Spectral Graph Convolution (ChebConv) trên ma trận Scaled Laplacian.
5. **STGCN** ([stgcn.py](file:///g:/nckh/stgcn.py)) [NEW]:
   * **Spatial**: Chebyshev Spectral Graph Convolution (ChebConv).
   * **Temporal**: Lớp Temporal Gated Convolution (GLU) qua tích chập 1D.
6. **STGCN-GCN** ([stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py)) [NEW]:
   * **Spatial**: Graph Convolutional Network (GCN) trích xuất đặc trưng không gian tĩnh đối xứng.
   * **Temporal**: Lớp Temporal Gated Convolution (GLU) tương tự như STGCN nhưng dùng GCN thay thế cho ChebNet.
7. **DCRNN** ([dcrnn.py](file:///g:/nckh/dcrnn.py)) [NEW]:
   * **Spatial**: Tích chập lan truyền (Diffusion Convolution) dựa trên bước đi ngẫu nhiên xuôi/ngược trên đồ thị có hướng.
   * **Temporal**: DCGRU (Diffusion Convolutional GRU Cell) theo luồng Encoder-Decoder.
8. **AGCRN** ([agcrn.py](file:///g:/nckh/agcrn.py)) [NEW]:
   * **Spatial**: NAP (Node Adaptive Parameter learning) sinh trọng số riêng cho từng nút đồ thị kết hợp với ma trận thích ứng tự học (AGCN) từ Node Embeddings.
   * **Temporal**: AGCRU Cell thay thế phép nhân GRU bằng AGCN.
9. **TGCN** ([tgcn.py](file:///g:/nckh/tgcn.py)) [NEW]:
   * **Spatial**: Graph Convolution chuẩn hóa đối xứng đối với ma trận kề cố định.
   * **Temporal**: TGCNCell tích hợp GCN đối xứng vào cổng GRU Cell.

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
python gat_tcn.py
python astgcn.py
python stgcn.py
python stgcn_gcn.py
python dcrnn.py
python agcrn.py
python tgcn.py
```
*Mỗi file mô hình đều được trang bị early stopping với độ kiên nhẫn `PATIENCE = 20`, thanh tiến trình `tqdm.auto` và tự động lưu checkpoint đạt MAE tốt nhất trên tập Validation.*

### 2. Chạy So sánh Đồng thời cả 9 Mô hình (compare_models.py)
Để thực hiện thực nghiệm so sánh tất cả các kiến trúc trên cùng một tập dữ liệu dùng chung (cùng cách phân chia Train/Val/Test), hãy sử dụng script [compare_models.py](file:///g:/nckh/compare_models.py):

* **Chế độ Đánh giá nhanh (Mặc định)**: Tải các checkpoint `.pth` tốt nhất của cả 9 mô hình hiện có và đánh giá trên tập Test để xuất bảng so sánh chỉ số:
  ```bash
  python compare_models.py --mode eval
  ```
* **Chế độ Huấn luyện mới**: Huấn luyện tuần tự cả 9 mô hình từ đầu, tự động lưu checkpoint và tổng hợp bảng so sánh:
  ```bash
  python compare_models.py --mode train --epochs 100
  ```

Sau khi chạy xong, kết quả so sánh (các chỉ số **Loss**, **MAE**, **MSE**, **RMSE** trên tập Test) sẽ được hiển thị trên console dưới dạng bảng Markdown và được tự động ghi vào tệp báo cáo `comparison_report.md`.

---

## 📝 Quy ước trong Nghiên cứu khoa học
* **Độ chia dữ liệu**: Mặc định chia theo tỉ lệ thời gian `80% Train` / `10% Val` / `10% Test`.
* **Khung thời gian dự báo**: Đầu vào `T_IN` sử dụng dữ liệu lịch sử của 120 phút trước (tương đương 24 bước với bước thời gian 5 phút), dự báo trước `HORIZON = 6` bước tiếp theo (30 phút tương lai).
