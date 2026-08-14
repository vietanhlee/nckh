import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from hybrid import Config
from stgcn import load_timeseries_double_rolling, load_adj_from_excel

def get_mape(y_true, y_pred):
    mask = y_true > 0.5
    if np.sum(mask) == 0: return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask])

def run_naive_baselines():
    print("🚀 BẮT ĐẦU CHẠY NAIVE BASELINES (Historical Average & Linear Regression)\n")
    cfg = Config()
    cfg.ROOT_DIR = "/workspace/GRAPH"
    cfg.ADJ_PATH = os.path.join(cfg.ROOT_DIR, "Graph_fix_py_3.xlsx")
    cfg.CSV_PATH = os.path.join(cfg.ROOT_DIR, "count_7_7_merg_sort_fix_fill.csv")

    _, nodes = load_adj_from_excel(cfg.ADJ_PATH)
    df_all = load_timeseries_double_rolling(
        cfg.CSV_PATH, nodes, cfg.DATA_WINDOW1, cfg.DATA_WINDOW2, cfg.TIME_STEP_MINUTES
    )

    # 1. Chia dữ liệu theo đúng tỷ lệ 80-10-10
    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[n_train + n_val:]
    
    print(f"Dataset Size - Train: {len(df_train)} | Test: {len(df_test)}")

    # 2. Xây dựng Historical Average (HA)
    # Trung bình lượng xe theo đúng Khung giờ trong ngày (Hour & Minute) từ tập Train
    print("\n--- Đang chạy Historical Average (HA) ---")
    df_train_copy = df_train.copy()
    df_train_copy['time_key'] = df_train_copy.index.strftime('%H:%M')
    
    # Bảng tra cứu HA cho từng node & feature theo time_key
    ha_lookup = df_train_copy.groupby('time_key').mean()
    
    # Đánh giá HA trên tập Test
    test_time_keys = df_test.index.strftime('%H:%M')
    
    # Dự báo cho toàn bộ Horizon (t+1 đến t+6)
    # Với HA, dự báo tại mốc t+h chính là giá trị trung bình tại thời điểm t+h trong quá khứ.
    ha_preds = []
    y_trues = []
    
    for h in range(1, cfg.HORIZON + 1):
        # Shift index để lấy time_key tương ứng với tương lai
        shifted_times = (df_test.index + pd.Timedelta(minutes=h*cfg.TIME_STEP_MINUTES)).strftime('%H:%M')
        
        # Mapping từ Lookup Table
        # Xử lý các time_key chưa từng thấy bằng giá trị trung bình toàn cục
        global_mean = ha_lookup.mean()
        pred_h = []
        for tk in shifted_times:
            if tk in ha_lookup.index:
                pred_h.append(ha_lookup.loc[tk].values)
            else:
                pred_h.append(global_mean.values)
                
        pred_h = np.array(pred_h) # (Samples, N*2)
        ha_preds.append(pred_h)
        
        # Ground Truth tương ứng
        if h < cfg.HORIZON:
            y_trues.append(df_test.iloc[h:-(cfg.HORIZON-h)].values)
        else:
            y_trues.append(df_test.iloc[h:].values)
            
    # Lấy mốc chung để tính Overall MAE (Bỏ phần râu ria do Horizon shift)
    # Chọn đoạn cắt ngắn nhất (đủ HORIZON)
    valid_len = len(df_test) - cfg.HORIZON
    ha_preds_stack = np.stack([p[:valid_len] for p in ha_preds], axis=1) # (Samples, Horizon, N*2)
    y_trues_stack = np.stack([df_test.iloc[i+1 : i+1+cfg.HORIZON].values for i in range(valid_len)], axis=0) # (Samples, Horizon, N*2)
    
    ha_mae = mean_absolute_error(y_trues_stack, ha_preds_stack)
    ha_rmse = np.sqrt(mean_squared_error(y_trues_stack, ha_preds_stack))
    
    # Tính tổng lượng xe để ra MAPE
    y_true_total = y_trues_stack.reshape(*y_trues_stack.shape[:-1], len(nodes), 2).sum(axis=-1)
    ha_pred_total = ha_preds_stack.reshape(*ha_preds_stack.shape[:-1], len(nodes), 2).sum(axis=-1)
    ha_mape = get_mape(y_true_total, ha_pred_total)
    
    print(f"✅ Historical Average (HA) -> MAE: {ha_mae:.4f} | RMSE: {ha_rmse:.4f} | MAPE: {ha_mape*100:.2f}%")

    # 3. Xây dựng Linear Regression / Ridge (Multi-Output)
    # Lấy T_in bước làm feature để dự báo HORIZON bước
    print("\n--- Đang chạy Linear Regression (Ridge) ---")
    def create_xy_dataset(df, T_in, Horizon):
        X, Y = [], []
        vals = df.values
        for i in range(len(vals) - T_in - Horizon + 1):
            X.append(vals[i : i+T_in].flatten())
            Y.append(vals[i+T_in : i+T_in+Horizon].flatten())
        return np.array(X), np.array(Y)

    X_train, Y_train = create_xy_dataset(df_train, cfg.T_IN, cfg.HORIZON)
    X_test, Y_test = create_xy_dataset(df_test, cfg.T_IN, cfg.HORIZON)
    
    # Chạy mô hình hồi quy Ridge (Linear Regression có L2 Regularization chống overfit)
    model = Ridge(alpha=1.0)
    model.fit(X_train, Y_train)
    lr_preds = model.predict(X_test)
    
    lr_mae = mean_absolute_error(Y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(Y_test, lr_preds))
    
    # Tính MAPE
    y_test_3d = Y_test.reshape(-1, cfg.HORIZON, len(nodes), 2)
    lr_preds_3d = lr_preds.reshape(-1, cfg.HORIZON, len(nodes), 2)
    
    lr_y_true_total = y_test_3d.sum(axis=-1)
    lr_pred_total = lr_preds_3d.sum(axis=-1)
    lr_mape = get_mape(lr_y_true_total, lr_pred_total)
    
    print(f"✅ Linear Regression (LR) -> MAE: {lr_mae:.4f} | RMSE: {lr_rmse:.4f} | MAPE: {lr_mape*100:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"🎉 TỔNG KẾT NAIVE BASELINES")
    print(f"{'='*50}")
    print(f"| Model | MAE | RMSE | MAPE (%) |")
    print(f"|-------|-----|------|----------|")
    print(f"| HA    | {ha_mae:.4f} | {ha_rmse:.4f} | {ha_mape*100:.2f}% |")
    print(f"| LR    | {lr_mae:.4f} | {lr_rmse:.4f} | {lr_mape*100:.2f}% |")
    print(f"{'='*50}")
    
    with open("naive_baselines_report.md", "w", encoding="utf-8") as f:
        f.write("# Kết quả Naive Baselines (Giai đoạn 2)\n\n")
        f.write("Dùng để điền vào bảng so sánh tổng thể (Table VIII).\n\n")
        f.write("| Model | MAE Overall | RMSE | MAPE (%) |\n")
        f.write("|-------|-------------|------|----------|\n")
        f.write(f"| Historical Average (HA) | {ha_mae:.4f} | {ha_rmse:.4f} | {ha_mape*100:.2f}% |\n")
        f.write(f"| Linear Regression (LR) | {lr_mae:.4f} | {lr_rmse:.4f} | {lr_mape*100:.2f}% |\n")

if __name__ == "__main__":
    run_naive_baselines()
