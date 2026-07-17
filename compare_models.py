import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse

# Import các lớp mô hình và cấu hình của cả 9 mô hình đang hoạt động
from gcn_lstm import ImprovedGNN_LSTM, Config as GCNLSTMConfig
from wavenet_gcn import GraphWaveNet_Model, Config as WaveNetConfig
from astgcn import ASTGCN_Model, Config as ASTGCNConfig
from stgcn import STGCN_Model, Config as STGCNConfig
from dcrnn import DCRNN_Model, Config as DCRNNConfig
from stgcn_gcn import STGCN_GCN_Model, Config as STGCN_GCNConfig
from stgcn_bilstm import STGCN_BiLSTM_Model, Config as STGCNBiLSTMConfig
from dcrnn_glu import DCRNN_GLU_Model, Config as DCRNNGLUConfig
from wavenet_glu import GraphWaveNet_GLU_Model, Config as WaveNetGLUConfig

# Tái sử dụng các hàm tiện ích nạp dữ liệu và đánh giá từ astgcn.py
from astgcn import (
    load_adj_from_excel,
    normalize_adj_sym,
    compute_scaled_laplacian,
    load_timeseries_double_rolling,
    MultiStepDataset,
    PureHuberLoss,
    train_one_epoch,
    evaluate
)

def main():
    parser = argparse.ArgumentParser(description="So sánh kết quả huấn luyện 9 mô hình Spatial-Temporal Graph NCKH.")
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'],
                        help="Chế độ chạy: 'train' (huấn luyện mới cả 9 mô hình từ đầu rồi so sánh) hoặc 'eval' (chỉ tải checkpoint và đánh giá).")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Số lượng epochs chạy thử nghiệm nếu chọn chế độ 'train' (mặc định lấy theo Config của từng mô hình).")
    args = parser.parse_args()

    # Khởi tạo instance của Config cho cả 9 mô hình để truy cập các properties (T_IN, FULL_SAVE_PATH, v.v.)
    gcn_lstm_cfg = GCNLSTMConfig()
    wavenet_cfg = WaveNetConfig()
    astgcn_cfg = ASTGCNConfig()
    stgcn_cfg = STGCNConfig()
    dcrnn_cfg = DCRNNConfig()
    stgcn_gcn_cfg = STGCN_GCNConfig()
    stgcn_bilstm_cfg = STGCNBiLSTMConfig()
    dcrnn_glu_cfg = DCRNNGLUConfig()
    wavenet_glu_cfg = WaveNetGLUConfig()

    # Đồng bộ hóa cấu hình SAVE_DIR sang thư mục tương đối cục bộ "model/"
    for cfg_inst in [gcn_lstm_cfg, wavenet_cfg, astgcn_cfg, stgcn_cfg, dcrnn_cfg, stgcn_gcn_cfg, stgcn_bilstm_cfg, dcrnn_glu_cfg, wavenet_glu_cfg]:
        cfg_inst.SAVE_DIR = "model/"
        os.makedirs(cfg_inst.SAVE_DIR, exist_ok=True)

    # Sử dụng config của GCN-LSTM làm cấu hình dữ liệu cơ bản
    cfg = gcn_lstm_cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM SO SÁNH CẢ 9 MÔ HÌNH")
    print(f"   Device: {device}")
    print(f"   Chế độ: {args.mode.upper()}")
    print(f"============================================================")

    # 1. Nạp dữ liệu và các ma trận đồ thị tương ứng
    print("\n[Bước 1] Nạp thông tin đồ thị và tiền xử lý dữ liệu...")
    A_raw, nodes = load_adj_from_excel(cfg.ADJ_PATH)
    A_norm = normalize_adj_sym(A_raw)
    L_tilde = compute_scaled_laplacian(A_raw)
    
    print(f"   - Số lượng nodes: {len(nodes)}")
    print(f"   - A_raw shape: {A_raw.shape}")
    print(f"   - A_norm shape: {A_norm.shape}")
    print(f"   - L_tilde shape: {L_tilde.shape}")

    print(f"\n   Nạp dữ liệu thời gian thực từ: {cfg.CSV_PATH}...")
    df_all = load_timeseries_double_rolling(
        cfg.CSV_PATH, nodes, cfg.DATA_WINDOW1, cfg.DATA_WINDOW2, cfg.TIME_STEP_MINUTES
    )

    n_total = len(df_all)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    idx_train_end = n_train
    idx_val_end = n_train + n_val

    df_train = df_all.iloc[:idx_train_end]
    df_val = df_all.iloc[idx_train_end:idx_val_end]
    df_test = df_all.iloc[idx_val_end:]

    # Khởi tạo Datasets dùng chung
    train_ds = MultiStepDataset(df_train, nodes, cfg.T_IN, cfg.HORIZON)
    scaler = {'mean': train_ds.means, 'std': train_ds.stds}
    val_ds = MultiStepDataset(df_val, nodes, cfg.T_IN, cfg.HORIZON, scaler)
    test_ds = MultiStepDataset(df_test, nodes, cfg.T_IN, cfg.HORIZON, scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE)

    print(f"   - Kích thước tập dữ liệu: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 2. Định nghĩa danh sách 9 mô hình sử dụng các Config instances tương ứng
    models_dict = {
        'GCN-LSTM': {
            'class': ImprovedGNN_LSTM,
            'config': gcn_lstm_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'gcn_hidden': gcn_lstm_cfg.GCN_HIDDEN,
                'lstm_hidden': gcn_lstm_cfg.LSTM_HIDDEN,
                'lstm_layers': gcn_lstm_cfg.LSTM_LAYERS,
                'output_feat': 1,
                'horizon': gcn_lstm_cfg.HORIZON,
                'A_norm': A_norm,
                'dropout': gcn_lstm_cfg.DROPOUT
            }
        },
        'Graph WaveNet': {
            'class': GraphWaveNet_Model,
            'config': wavenet_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'residual_channels': wavenet_cfg.RESIDUAL_CHANNELS,
                'skip_channels': wavenet_cfg.SKIP_CHANNELS,
                'dilation_list': wavenet_cfg.DILATION_LIST,
                'adaptive_emb_dim': wavenet_cfg.ADAPTIVE_EMB,
                'horizon': wavenet_cfg.HORIZON,
                'output_feat': 1,
                'A_norm': A_norm,
                'dropout': wavenet_cfg.DROPOUT
            }
        },
        'Graph WaveNet-GLU': {
            'class': GraphWaveNet_GLU_Model,
            'config': wavenet_glu_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'residual_channels': wavenet_glu_cfg.RESIDUAL_CHANNELS,
                'skip_channels': wavenet_glu_cfg.SKIP_CHANNELS,
                'dilation_list': wavenet_glu_cfg.DILATION_LIST,
                'adaptive_emb_dim': wavenet_glu_cfg.ADAPTIVE_EMB,
                'horizon': wavenet_glu_cfg.HORIZON,
                'output_feat': 1,
                'A_norm': A_norm,
                'dropout': wavenet_glu_cfg.DROPOUT
            }
        },
        'ASTGCN': {
            'class': ASTGCN_Model,
            'config': astgcn_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': astgcn_cfg.BLOCK_HIDDEN,
                'num_blocks': astgcn_cfg.NUM_BLOCKS,
                'T_in': astgcn_cfg.T_IN,
                'cheb_K': astgcn_cfg.CHEB_K,
                'horizon': astgcn_cfg.HORIZON,
                'output_feat': 1,
                'L_tilde': L_tilde,
                'dropout': astgcn_cfg.DROPOUT
            }
        },
        'STGCN': {
            'class': STGCN_Model,
            'config': stgcn_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': stgcn_cfg.BLOCK_HIDDEN,
                'num_blocks': stgcn_cfg.NUM_BLOCKS,
                'T_in': stgcn_cfg.T_IN,
                'cheb_K': stgcn_cfg.CHEB_K,
                'horizon': stgcn_cfg.HORIZON,
                'output_feat': 1,
                'L_tilde': L_tilde,
                'dropout': stgcn_cfg.DROPOUT
            }
        },
        'STGCN-GCN': {
            'class': STGCN_GCN_Model,
            'config': stgcn_gcn_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': stgcn_gcn_cfg.BLOCK_HIDDEN,
                'num_blocks': stgcn_gcn_cfg.NUM_BLOCKS,
                'T_in': stgcn_gcn_cfg.T_IN,
                'horizon': stgcn_gcn_cfg.HORIZON,
                'output_feat': 1,
                'A_norm': A_norm,
                'dropout': stgcn_gcn_cfg.DROPOUT
            }
        },
        'STGCN-BiLSTM': {
            'class': STGCN_BiLSTM_Model,
            'config': stgcn_bilstm_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': stgcn_bilstm_cfg.BLOCK_HIDDEN,
                'num_blocks': stgcn_bilstm_cfg.NUM_BLOCKS,
                'T_in': stgcn_bilstm_cfg.T_IN,
                'cheb_K': stgcn_bilstm_cfg.CHEB_K,
                'horizon': stgcn_bilstm_cfg.HORIZON,
                'output_feat': 1,
                'L_tilde': L_tilde,
                'dropout': stgcn_bilstm_cfg.DROPOUT
            }
        },
        'DCRNN': {
            'class': DCRNN_Model,
            'config': dcrnn_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'hidden_dim': dcrnn_cfg.HIDDEN_DIM,
                'K': dcrnn_cfg.K,
                'horizon': dcrnn_cfg.HORIZON,
                'output_feat': 1,
                'A_raw': A_raw,
                'dropout': dcrnn_cfg.DROPOUT
            }
        },
        'DCRNN-GLU': {
            'class': DCRNN_GLU_Model,
            'config': dcrnn_glu_cfg,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': dcrnn_glu_cfg.BLOCK_HIDDEN,
                'num_blocks': dcrnn_glu_cfg.NUM_BLOCKS,
                'T_in': dcrnn_glu_cfg.T_IN,
                'K': dcrnn_glu_cfg.K,
                'horizon': dcrnn_glu_cfg.HORIZON,
                'output_feat': 1,
                'A_raw': A_raw,
                'dropout': dcrnn_glu_cfg.DROPOUT
            }
        }
    }

    results = {}

    # 3. Tiến hành Huấn luyện hoặc Đánh giá
    for model_name, model_info in models_dict.items():
        print(f"\n==========================================")
        print(f"⚙️ Đang xử lý mô hình: {model_name}")
        print(f"==========================================")
        
        # Khởi tạo mô hình
        model = model_info['class'](**model_info['args']).to(device)
        save_path = model_info['config'].FULL_SAVE_PATH

        if args.mode == 'train':
            print(f"-> Bắt đầu huấn luyện mô hình {model_name}...")
            epochs = args.epochs if args.epochs is not None else model_info['config'].EPOCHS
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_info['config'].LEARNING_RATE)
            loss_fn = PureHuberLoss(delta=model_info['config'].LOSS_DELTA)
            grad_scaler = torch.amp.GradScaler('cuda')

            best_mae = float('inf')
            patience = model_info['config'].PATIENCE
            patience_cnt = 0

            for ep in range(epochs):
                train_loss, train_mae = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, device, grad_scaler, scaler
                )
                val_metrics = evaluate(model, val_loader, device, scaler, loss_fn=loss_fn, verbose=False)
                val_mae = val_metrics['mae']
                val_loss = val_metrics['loss']

                print(f"Ep {ep+1:03d} | Loss: {train_loss:.4f} / {val_loss:.4f} | MAE: {train_mae:.2f} / {val_mae:.2f}", end="")

                if val_mae < best_mae:
                    best_mae = val_mae
                    patience_cnt = 0
                    torch.save(model.state_dict(), save_path)
                    print(" -> Saved Best")
                else:
                    patience_cnt += 1
                    print(f" | Patience: {patience_cnt}/{patience}")
                    if patience_cnt >= patience:
                        print("Early Stopping")
                        break

        # Đánh giá trên tập test sử dụng checkpoint tốt nhất
        print(f"-> Đánh giá mô hình {model_name} trên tập TEST...")
        if not os.path.exists(save_path):
            print(f"❌ KHÔNG tìm thấy checkpoint tại {save_path}. Bỏ qua đánh giá.")
            continue

        model.load_state_dict(torch.load(save_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device, scaler, loss_fn=PureHuberLoss(delta=cfg.LOSS_DELTA))
        
        results[model_name] = test_metrics
        print(f"   [Test Results] Loss: {test_metrics['loss']:.4f} | MAE: {test_metrics['mae']:.4f} | MSE: {test_metrics['mse']:.4f} | RMSE: {test_metrics['rmse']:.4f}")

        # Dọn dẹp GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Xuất báo cáo so sánh kết quả dạng bảng Markdown
    if len(results) == 0:
        print("\n❌ Không có kết quả mô hình nào được đánh giá thành công.")
        return

    print(f"\n============================================================")
    print(f"📊 BẢNG SO SÁNH KẾT QUẢ TRÊN TẬP TEST")
    print(f"============================================================")
    
    report_data = []
    for model_name, metrics in results.items():
        report_data.append({
            'Model': model_name,
            'Test Loss (Huber)': f"{metrics['loss']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'MSE': f"{metrics['mse']:.4f}",
            'RMSE': f"{metrics['rmse']:.4f}"
        })
    
    df_report = pd.DataFrame(report_data)
    markdown_table = df_report.to_markdown(index=False)
    print(markdown_table)
    print(f"============================================================")

    # Lưu bảng so sánh vào file markdown
    report_path = "comparison_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Báo cáo so sánh các mô hình Spatial-Temporal Graph (NCKH)\n\n")
            f.write(f"Chế độ chạy thực nghiệm: **{args.mode.upper()}**\n\n")
            f.write(markdown_table)
            f.write("\n")
        print(f"💾 Đã lưu báo cáo so sánh vào: {report_path}")
    except Exception as e:
        print(f"⚠️ Không thể lưu báo cáo ra file: {e}")

if __name__ == "__main__":
    main()
