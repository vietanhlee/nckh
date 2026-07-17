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

### 5. **STGCN-GCN** (STGCN using standard GCN instead of ChebNet) [NEW]
* **Tệp**: [stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py) [NEW]
* **Ý tưởng**:
  * Biến thể của STGCN sử dụng lớp tích chập không gian GCN thông thường dựa trên ma trận kề chuẩn hóa đối xứng $A_{norm}$ thay thế cho Chebyshev Spectral Graph Conv (ChebConv) trên ma trận Scaled Laplacian $L_{tilde}$.
  * Giữ nguyên cấu trúc Temporal Conv (GLU) ở hai bên và LayerNorm để so sánh rõ ràng sự khác biệt về mặt hiệu quả giữa tích chập phổ (Spectral Conv) và tích chập không gian thông thường (Spatial Conv).

---

## 📅 Các tệp đã thay đổi cụ thể

### [New Models]
* #### [NEW] [stgcn.py](file:///g:/nckh/stgcn.py)
* #### [NEW] [dcrnn.py](file:///g:/nckh/dcrnn.py)
* #### [NEW] [agcrn.py](file:///g:/nckh/agcrn.py)
* #### [NEW] [tgcn.py](file:///g:/nckh/tgcn.py)
* #### [NEW] [stgcn_gcn.py](file:///g:/nckh/stgcn_gcn.py)

### [Comparison & Documentation]
* #### [MODIFY] [compare_models.py](file:///g:/nckh/compare_models.py)
  Tích hợp các mô hình thành **9 mô hình** so sánh tổng thể (loại bỏ GCN-TCN và ASTGCN-GCN).
* #### [MODIFY] [README.md](file:///g:/nckh/README.md)
  Bổ sung mô tả và cách chạy của mô hình mới.

---

## 🔍 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra tự động
- Chạy thử 1 epoch huấn luyện của mô hình STGCN-GCN riêng lẻ để kiểm tra độ chính xác cú pháp.
- Chạy thử `python compare_models.py --mode train --epochs 1` để kiểm tra khả năng chạy tuần tự và tổng hợp kết quả của cả **9 mô hình**.

### Kiểm tra thủ công
- Xác nhận bảng so sánh kết quả in ra terminal hiển thị đầy đủ thông tin của 9 mô hình.
- Xác nhận báo cáo `comparison_report.md` được ghi nhận chính xác.
