# 📑 Báo cáo Chi tiết Kiến trúc 3 Mô hình Dự báo Lưu lượng Giao thông

Báo cáo này mô tả sơ đồ kiến trúc, luồng dữ liệu và cấu hình của **3 mô hình** trong dự án Nghiên cứu Khoa học:
1. **GCN-LSTM** ([gcn_lstm.py](file:///g:/nckh/gcn_lstm.py))
2. **STGCN GLU-ATTN (Hybrid)** ([hybrid.py](file:///g:/nckh/hybrid.py))
3. **STGCN Pure Attn (Block-Attn)** ([stgcn_block_attn.py](file:///g:/nckh/stgcn_block_attn.py))

---

## 📊 1. Bảng So sánh Tổng quan 3 Mô hình

| Tiêu chí | GCN-LSTM | STGCN GLU-ATTN (Hybrid) | STGCN Pure Attn (Block-Attn) |
| :--- | :--- | :--- | :--- |
| **Mã nguồn** | [gcn_lstm.py](file:///g:/nckh/gcn_lstm.py) | [hybrid.py](file:///g:/nckh/hybrid.py) | [stgcn_block_attn.py](file:///g:/nckh/stgcn_block_attn.py) |
| **Học Không gian (Spatial)** | GCN 2 tầng tĩnh | ChebNet Phổ ($K=3$) | ChebNet Phổ ($K=3$) |
| **Học Thời gian (Temporal)** | LSTM | Gated 1D Conv (GLU) | Multi-Head Temporal Self-Attention |
| **Cấu trúc trong Block** | GCN -> LSTM | GLU -> ChebNet -> GLU | Attn1 -> ChebNet -> Attn2 |
| **Vị trí Attention** | Không có | Cuối mô hình (Model-Level) | Trong từng Block & Cuối mô hình |
| **Số lớp Attention** | 0 | 1 lớp | **5 lớp** (2/block × 2 blocks + 1 final) |
| **Số Blocks (`NUM_BLOCKS`)** | N/A | **2 Blocks** | **2 Blocks** |
| **Số Channels (`Hidden`)** | GCN=32, LSTM=64 | Block=64 | Block=64 (Heads=4) |

---

## 🏛️ 2. Mô hình 1: GCN-LSTM

### Mô tả
Kết hợp **GCN tĩnh** và **LSTM**. Ở mỗi bước thời gian $t$, GCN trích xuất đặc trưng không gian giữa các nút. Sau đó, chuỗi đặc trưng được đưa qua LSTM để học phụ thuộc thời gian.

### Sơ đồ Luồng xử lý (Architecture Flowchart)

```text
┌─────────────────────────────────────────────────────────┐
│ Input X (Shape: Batch, 24 steps, 557 nodes, 4 features) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │ 1. Reshape & GCN Layer 1 + ReLU + Dropout             │
 ├───────────────────────────────────────────────────────┤
 │ 2. GCN Layer 2 + ReLU + Dropout                       │
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │ 3. Reshape & LSTM Layer (Hidden = 64)                 │
 ├───────────────────────────────────────────────────────┤
 │ 4. Extract Last Hidden State (tại bước t = 24)        │
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │ 5. MLP Shared (LayerNorm + ReLU + Dropout)            │
 ├───────────────────────────────────────────────────────┤
 │ 6. Linear Projection                                  │
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Output Y (Shape: Batch, 6 steps, 557 nodes, 1 feature)  │
└─────────────────────────────────────────────────────────┘
```

---

## 🏛️ 3. Mô hình 2: STGCN GLU-ATTN (Hybrid)

### Mô tả
Gồm **2 khối STGCN Block** (dùng Gated 1D Conv GLU bắt thời gian ngắn hạn và ChebNet $K=3$ học không gian), kết hợp **1 lớp Multi-Head Temporal Self-Attention ở CUỐI MÔ HÌNH** trước khi chiếu ra đầu ra.

### Sơ đồ Luồng xử lý (Architecture Flowchart)

```text
┌─────────────────────────────────────────────────────────┐
│ Input X (Shape: Batch, 24 steps, 557 nodes, 4 features) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STGCN BLOCK 1                                           │
│  ├─ Temporal Conv GLU 1                                 │
│  ├─ Spatial Graph Conv (ChebNet K=3) + ReLU             │
│  ├─ Temporal Conv GLU 2                                 │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ STGCN BLOCK 2                                           │
│  ├─ Temporal Conv GLU 1                                 │
│  ├─ Spatial Graph Conv (ChebNet K=3) + ReLU             │
│  ├─ Temporal Conv GLU 2                                 │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ FINAL TEMPORAL SELF-ATTENTION (Model-Level Attention)   │
│  └─ Multi-Head Attention + Residual + LayerNorm + FFN   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ CONV 1D TIME REDUCTION (Nắn chiều thời gian 24 -> 1)   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Output Y (Shape: Batch, 6 steps, 557 nodes, 1 feature)  │
└────────────────────────────┬────────────────────────────┘
```

---

## 🏛️ 4. Mô hình 3: STGCN Pure Attn (Block-Attn)

### Mô tả
Kiến trúc **Pure Spatio-Temporal Graph Attention Transformer** (dùng **2 Blocks**). Toàn bộ các lớp GLU cục bộ đều được thay bằng **Multi-Head Temporal Self-Attention** ở cả 2 vị trí trong từng khối `STGCNBlockAttn`, kết hợp thêm **1 lớp Final Temporal Attention ở cuối** (Hierarchical Dual Attention).

### Sơ đồ Luồng xử lý (Architecture Flowchart)

```text
┌─────────────────────────────────────────────────────────┐
│ Input X (Shape: Batch, 24 steps, 557 nodes, 4 features) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ LINEAR IN-PROJECTION (Conv 1x1 nắn channels -> 64)       │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ BLOCK 1 (Pure Attention Block)                          │
│  ├─ Temporal Self-Attention 1 (Multi-Head + FFN)        │
│  ├─ Spatial Graph Conv (ChebNet K=3) + ReLU             │
│  ├─ Temporal Self-Attention 2 (Multi-Head + FFN)        │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ BLOCK 2 (Pure Attention Block)                          │
│  ├─ Temporal Self-Attention 1 (Multi-Head + FFN)        │
│  ├─ Spatial Graph Conv (ChebNet K=3) + ReLU             │
│  ├─ Temporal Self-Attention 2 (Multi-Head + FFN)        │
│  └─ Residual Add + LayerNorm + Dropout                  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ FINAL TEMPORAL SELF-ATTENTION (Hierarchical Dual)       │
│  └─ Multi-Head Attention + Residual + LayerNorm + FFN   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ CONV 1D TIME REDUCTION (Nắn chiều thời gian 24 -> 1)   │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│ Output Y (Shape: Batch, 6 steps, 557 nodes, 1 feature)  │
└────────────────────────────┬────────────────────────────┘
```

---

## ⚙️ 5. Chi tiết Cấu hình Hyperparameters

| Tham số | GCN-LSTM | STGCN GLU-ATTN | STGCN Pure Attn |
| :--- | :---: | :---: | :---: |
| `BATCH_SIZE` | 32 | 64 | 32 |
| `LEARNING_RATE` | 0.001 | 0.0005 | 0.0005 |
| `PATIENCE` | 40 | 60 | 60 |
| `NUM_BLOCKS` | N/A | **2** | **2** |
| `BLOCK_HIDDEN` | GCN=32, LSTM=64 | 64 | 64 |
| `CHEB_K` | N/A | 3 | 3 |
| `ATTN_NUM_HEADS` | N/A | 4 | 4 |
| `LOSS_DELTA` | 1.0 | 1.0 | 1.0 |
| `USE_EMA` | False | True (0.995) | True (0.995) |
| `USE_LR_SCHEDULER` | False | True | True |
