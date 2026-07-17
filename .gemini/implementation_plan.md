# Kế hoạch Thực hiện: Bổ sung 4 mô hình mới (STGCN, DCRNN, AGCRN, TGCN) vào so sánh thực nghiệm (NCKH)

Để làm phong phú thêm kết quả thực nghiệm NCKH của bạn, chúng ta sẽ xây dựng thêm 4 mô hình Spatial-Temporal Graph mới bằng PyTorch thuần (để tránh xung đột thư viện ngoài như PyTorch Geometric hay DGL) và tích hợp vào script so sánh `compare_models.py`.

---

## 🛠️ Thiết kế Kiến trúc 4 Mô hình Mới

### 1. **STGCN** (Spatio-Temporal Graph Convolutional Networks)
* **Tệp**: [stgcn.py](file:///g:/nckh/stgcn.py) [NEW]
* **Ý tưởng**:
  * Chồng nhiều khối **ST-Conv block**. Mỗi block gồm: 1 lớp Temporal Gated Conv (sử dụng 1D Causal Conv kết hợp với Gated Linear Unit - GLU) -> 1 lớp Spatial Graph Conv (sử dụng đa thức Chebyshev trên ma trận L_tilde tương tự ASTGCN) -> 1 lớp Temporal Gated Conv (GLU).
  * Đầu ra đi qua lớp tích chập cuối để chiếu ra số bước dự báo `Horizon`.

### 2. **DCRNN** (Diffusion Convolutional Recurrent Neural Network)
* **Tệp**: [dcrnn.py](file:///g:/nckh/dcrnn.py) [NEW]
* **Ý tưởng**:
  * Thực hiện tích chập lan truyền (Diffusion Convolution) bằng tổng các bước lan truyền ngẫu nhiên (random walk) xuôi và ngược trên đồ thị:
    $$H = \sum_{k=0}^{K} \left( W_{k,1} (D_O^{-1} A)^k X + W_{k,2} (D_I^{-1} A^T)^k X \right)$$
  * Tích hợp Diffusion Conv vào các cổng của GRU tạo thành tế bào `DCGRUCell`.
  * Xây dựng luồng Encoder-Decoder tuần tự qua các bước thời gian để sinh dự báo `Horizon` bước.

### 3. **AGCRN** (Adaptive Graph Convolutional Recurrent Network)
* **Tệp**: [agcrn.py](file:///g:/nckh/agcrn.py) [NEW]
* **Ý tưởng**:
  * **NAP (Node Adaptive Parameter learning)**: Tự động học ma trận đặc trưng nút (Node Embeddings) để sinh tham số trọng số riêng biệt cho từng nút đồ thị.
  * **AGCN (Adaptive Graph Convolution)**: Tự sinh ma trận kề thích ứng động: $A_{adp} = Softmax(ReLU(E \cdot E^T))$ mà không cần ma trận kề tĩnh từ Excel.
  * **AGCRU**: Thay thế các phép tính tuyến tính trong GRU bằng tích chập thích ứng AGCN.

### 4. **TGCN** (Temporal Graph Convolutional Network)
* **Tệp**: [tgcn.py](file:///g:/nckh/tgcn.py) [NEW]
* **Ý tưởng**:
  * Kết hợp GCN đơn giản và GRU.
  * Định nghĩa lớp `TGCNCell`: Thay thế toàn bộ các phép toán nhân ma trận của GRU Cell bằng phép toán Graph Convolution sử dụng ma trận kề chuẩn hóa đối xứng $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$.
  * Dự báo trực tiếp ra `Horizon` thông qua lớp tuyến tính từ hidden state của timestep cuối cùng.

---

## 📅 Các tệp sẽ thay đổi cụ thể

### [New Models]
#### [NEW] [stgcn.py](file:///g:/nckh/stgcn.py)
#### [NEW] [dcrnn.py](file:///g:/nckh/dcrnn.py)
#### [NEW] [agcrn.py](file:///g:/nckh/agcrn.py)
#### [NEW] [tgcn.py](file:///g:/nckh/tgcn.py)

### [Comparison & Documentation]
#### [MODIFY] [compare_models.py](file:///g:/nckh/compare_models.py)
Import thêm 4 mô hình mới, định nghĩa tham số Config và tích hợp vào danh sách so sánh.
#### [MODIFY] [README.md](file:///g:/nckh/README.md)
Cập nhật mô tả kiến trúc và cách chạy của 4 mô hình mới vào tài liệu.

---

## 🔍 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra tự động
- Chạy thử 1 epoch huấn luyện của từng file trong 4 file mô hình mới (`python stgcn.py`, v.v.) để chắc chắn tqdm, GradScaler mới chạy mượt mà, không lỗi cú pháp.
- Chạy thử `python compare_models.py --mode train --epochs 1` để kiểm tra khả năng chạy tuần tự và tổng hợp kết quả của cả **8 mô hình** (4 cũ + 4 mới).

### Kiểm tra thủ công
- Xác nhận bảng so sánh kết quả in ra terminal hiển thị đầy đủ thông tin của 8 mô hình.
- Xác nhận báo cáo `comparison_report.md` được ghi nhận chính xác.
