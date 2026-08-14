# 🛡️ Báo cáo Phân tích Độ nhạy với Nhiễu Nhận dạng Cho Tất cả Model

Báo cáo giải quyết triệt để phản biện của Reviewer về **Vấn đề Lan truyền sai số (Error Propagation)** từ Giai đoạn 1 sang Giai đoạn 2.

## 📊 Bảng Kết quả Đánh giá Độ suy giảm Hiệu năng

| Model Architecture         |   Input Noise Level (ΔMAE) | Forecasting MAE (Mean ± Std)   |   Error Degradation (vs Clean) |
|:---------------------------|---------------------------:|:-------------------------------|-------------------------------:|
| STGCN (Baseline)           |                       0    | 3.3501 ± 0.0134                |                         0      |
| STGCN (Baseline)           |                       3.49 | 5.6996 ± 0.1303                |                         2.3495 |
| STGCN (Baseline)           |                       4.64 | 7.0910 ± 0.1349                |                         3.741  |
| STGCN (Baseline)           |                      10.25 | 12.3367 ± 0.5106               |                         8.9866 |
| STGCN (Baseline)           |                      13.95 | 13.3148 ± 0.6747               |                         9.9647 |
| GraphWaveNet               |                       0    | 3.3290 ± 0.0168                |                         0      |
| GraphWaveNet               |                       3.49 | 5.7778 ± 0.0929                |                         2.4489 |
| GraphWaveNet               |                       4.64 | 7.1724 ± 0.1051                |                         3.8434 |
| GraphWaveNet               |                      10.25 | 13.0573 ± 0.1779               |                         9.7283 |
| GraphWaveNet               |                      13.95 | 14.9194 ± 0.1783               |                        11.5904 |
| ASTGCN                     |                       0    | 3.3331 ± 0.0059                |                         0      |
| ASTGCN                     |                       3.49 | 5.7374 ± 0.0786                |                         2.4043 |
| ASTGCN                     |                       4.64 | 7.0689 ± 0.0927                |                         3.7358 |
| ASTGCN                     |                      10.25 | 12.3714 ± 0.2007               |                         9.0382 |
| ASTGCN                     |                      13.95 | 14.0552 ± 0.2814               |                        10.722  |
| STAEformer                 |                       0    | 3.2671 ± 0.0110                |                         0      |
| STAEformer                 |                       3.49 | 5.3737 ± 0.2013                |                         2.1066 |
| STAEformer                 |                       4.64 | 6.5295 ± 0.3338                |                         3.2624 |
| STAEformer                 |                      10.25 | 11.7051 ± 1.0125               |                         8.4381 |
| STAEformer                 |                      13.95 | 13.8404 ± 1.2862               |                        10.5733 |
| MegaCRN                    |                       0    | 3.3311 ± 0.0035                |                         0      |
| MegaCRN                    |                       3.49 | 5.5945 ± 0.0413                |                         2.2634 |
| MegaCRN                    |                       4.64 | 6.8725 ± 0.0584                |                         3.5414 |
| MegaCRN                    |                      10.25 | 12.3829 ± 0.1591               |                         9.0518 |
| MegaCRN                    |                      13.95 | 14.0384 ± 0.2132               |                        10.7073 |
| DSTAGNN                    |                       0    | 3.3565 ± 0.0483                |                         0      |
| DSTAGNN                    |                       3.49 | 6.2049 ± 0.1285                |                         2.8484 |
| DSTAGNN                    |                       4.64 | 7.6733 ± 0.1457                |                         4.3168 |
| DSTAGNN                    |                      10.25 | 12.7206 ± 0.6160               |                         9.3641 |
| DSTAGNN                    |                      13.95 | 13.6131 ± 1.3263               |                        10.2566 |
| TA-STGCN (Proposed / Ours) |                       0    | 3.3304 ± 0.0117                |                         0      |
| TA-STGCN (Proposed / Ours) |                       3.49 | 5.3865 ± 0.2251                |                         2.0561 |
| TA-STGCN (Proposed / Ours) |                       4.64 | 6.6007 ± 0.2844                |                         3.2703 |
| TA-STGCN (Proposed / Ours) |                      10.25 | 11.1374 ± 0.6686               |                         7.807  |
| TA-STGCN (Proposed / Ours) |                      13.95 | 11.9411 ± 0.8964               |                         8.6107 |

---
