# Kế hoạch Thực hiện: So sánh thực nghiệm 7 kiến trúc Spatial-Temporal Graph (NCKH)

Dự án nghiên cứu khoa học tập trung so sánh 7 kiến trúc đồ thị không-thời gian bằng PyTorch thuần để phục vụ dự báo lưu lượng giao thông, sử dụng thư mục checkpoint cục bộ để tăng tính linh hoạt.

---

## 🛠️ Thiết kế Kiến trúc các Mô hình

### 1. **GCN-LSTM** ([gcn_lstm.py](file:///g:/nckh/gcn_lstm.py))
* Kết hợp GCN trích xuất đặc trưng không gian tĩnh và LSTM học phụ thuộc thời gian.

### 2. **Graph WaveNet** ([wavenet_gcn.py](file:///g:/nckh/wavenet_gcn.py))
* Tích hợp ma trận kề thích ứng động tự học song song với ma trận kề tĩnh và Gated TCN (Dilated Causal Conv).

### 3. **ASTGCN** ([astgcn.py](file:///g:/nckh/astgcn.py))
* Cơ chế Spatial-Temporal Attention động cùng Chebyshev Spectral Graph Conv (ChebConv).

### 4. **STGCN** ([stgcn.py](file:///g:/nckh/stgcn.py)) [NEW]
* Chebyshev Spectral Graph Conv (ChebConv) kết hợp lớp Temporal Gated Conv (GLU) qua tích chập 1D.

### 5. **STGCN-GCN** ([stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py)) [NEW]
* Biến thể của STGCN sử dụng tích chập không gian GCN tĩnh thay vì ChebNet, giữ nguyên cấu trúc Temporal Conv (GLU).

### 6. **STGCN-BiLSTM** ([stgcn_bilstm.py](file:///g:/nckh/stgcn_bilstm.py)) [NEW - ĐỀ XUẤT]
* Biến thế STGCN thiết kế lại bộ phận học thời gian sử dụng Bidirectional LSTM (BiLSTM) thay thế cho lớp Temporal Gated Conv (GLU) gốc để tăng cường học đặc trưng chuỗi thời gian hai chiều.

### 7. **DCRNN** ([dcrnn.py](file:///g:/nckh/dcrnn.py)) [NEW]
* Tích chập lan truyền (Diffusion Convolution) dựa trên bước đi ngẫu nhiên trên đồ thị tích hợp vào DCGRU Cell theo cấu trúc Sequence-to-Sequence.

---

## 📅 Các tệp đã thay đổi cụ thể

### [New Models & Checkpoint path updates]
* #### [NEW] [stgcn.py](file:///g:/nckh/stgcn.py)
* #### [NEW] [dcrnn.py](file:///g:/nckh/dcrnn.py)
* #### [NEW] [stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py)
* #### [NEW] [stgcn_bilstm.py](file:///g:/nckh/stgcn_bilstm.py)
* #### [MODIFY] `gcn_lstm.py`, `wavenet_gcn.py`, `astgcn.py`, `stgcn.py`, `stgcn_gcn.py`, `stgcn_bilstm.py`, `dcrnn.py`
  Chuyển đổi tham số đường dẫn lưu mô hình `SAVE_DIR` sang thư mục tương đối `"model/"`.

### [Comparison & Documentation]
* #### [MODIFY] [compare_models.py](file:///g:/nckh/compare_models.py)
  Tích hợp các mô hình thành **7 mô hình** so sánh tổng thể và tự động tạo thư mục checkpoint `model/` nếu chưa có.
* #### [MODIFY] [README.md](file:///g:/nckh/README.md)
  Bổ sung mô tả và cách chạy của 7 mô hình.

---

## 🔍 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra tự động
- Chạy thử `python compare_models.py --mode train --epochs 1` để kiểm tra khả năng chạy tuần tự và tổng hợp kết quả của cả **7 mô hình**.

### Kiểm tra thủ công
- Xác nhận thư mục `./model/` được tự sinh tự động tại thư mục dự án và lưu trữ các file checkpoint `.pth` tốt nhất.
- Xác nhận bảng so sánh kết quả in ra terminal hiển thị đầy đủ thông tin của 7 mô hình.
- Xác nhận báo cáo `comparison_report.md` được ghi nhận chính xác.
