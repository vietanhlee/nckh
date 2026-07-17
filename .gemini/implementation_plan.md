# Kế hoạch Thực hiện: Bổ sung các mô hình mới vào so sánh thực nghiệm (NCKH)

Để làm phong phú thêm kết quả thực nghiệm NCKH của bạn, chúng ta sẽ xây dựng thêm các mô hình Spatial-Temporal Graph mới bằng PyTorch thuần (để tránh xung đột thư viện ngoài như PyTorch Geometric hay DGL) và tích hợp vào script so sánh `compare_models.py`.

---

## 🛠️ Thiết kế Kiến trúc các Mô hình Mới

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

### 5. **GCN-TCN** (Graph Convolutional Network + Temporal Convolutional Network) [NEW]
* **Tệp**: [gcn_tcn.py](file:///g:/nckh/gcn_tcn.py) [NEW]
* **Ý tưởng**:
  * Kết hợp GCN tĩnh và tích chập temporal dạng TCN thay vì LSTM hay GRU.
  * Trích xuất đặc trưng không gian tĩnh bằng 2 lớp GCN với ma trận kề đối xứng chuẩn hóa tĩnh.
  * Phụ thuộc thời gian được học bằng mạng TCN (stacked TCN block) với dilated causal conv 1D để tránh nhìn trước tương lai và tăng kích thước receptive field theo cấp số nhân.
  * Final projection chiếu kết quả về chiều rộng dự báo `Horizon`.

---

## 📅 Các tệp đã thay đổi cụ thể

### [New Models]
* #### [NEW] [stgcn.py](file:///g:/nckh/stgcn.py)
* #### [NEW] [dcrnn.py](file:///g:/nckh/dcrnn.py)
* #### [NEW] [agcrn.py](file:///g:/nckh/agcrn.py)
* #### [NEW] [tgcn.py](file:///g:/nckh/tgcn.py)
* #### [NEW] [gcn_tcn.py](file:///g:/nckh/gcn_tcn.py)

### [Comparison & Documentation]
* #### [MODIFY] [compare_models.py](file:///g:/nckh/compare_models.py)
  Tích hợp GCN-TCN thành **9 mô hình** so sánh tổng thể, cập nhật instance Config tương ứng.
* #### [MODIFY] [README.md](file:///g:/nckh/README.md)
  Bổ sung mô tả và cách chạy của mô hình GCN-TCN.

---

## 🔍 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra tự động
- Chạy thử 1 epoch huấn luyện của từng mô hình riêng lẻ (`python gcn_tcn.py`, v.v.) để kiểm tra độ chính xác cú pháp.
- Chạy thử `python compare_models.py --mode train --epochs 1` để kiểm tra khả năng chạy tuần tự và tổng hợp kết quả của cả **9 mô hình**.

### Kiểm tra thủ công
- Xác nhận bảng so sánh kết quả in ra terminal hiển thị đầy đủ thông tin của 9 mô hình.
- Xác nhận báo cáo `comparison_report.md` được ghi nhận chính xác.
