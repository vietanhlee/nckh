import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse

# Import các lớp mô hình và cấu hình của cả 8 mô hình
from gcn_lstm import ImprovedGNN_LSTM, Config as GCNLSTMConfig
from wavenet_gcn import GraphWaveNet_Model, Config as WaveNetConfig
from gat_tcn import GAT_TCN_Model, Config as GATConfig
from astgcn import ASTGCN_Model, Config as ASTGCNConfig
from stgcn import STGCN_Model, Config as STGCNConfig
from dcrnn import DCRNN_Model, Config as DCRNNConfig
from agcrn import AGCRN_Model, Config as AGCRNConfig
from tgcn import TGCN_Model, Config as TGCNConfig

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
    parser = argparse.ArgumentParser(description="So sánh kết quả huấn luyện 8 mô hình Spatial-Temporal Graph NCKH.")
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'],
                        help="Chế độ chạy: 'train' (huấn luyện mới cả 8 mô hình từ đầu rồi so sánh) hoặc 'eval' (chỉ tải checkpoint và đánh giá).")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Số lượng epochs chạy thử nghiệm nếu chọn chế độ 'train' (mặc định lấy theo Config của từng mô hình).")
    args = parser.parse_args()

    # Sử dụng config của GCN-LSTM làm cấu hình dữ liệu cơ bản
    cfg = GCNLSTMConfig
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f"🚀 BẮT ĐẦU CHẠY THỬ NGHIỆM SO SÁNH CẢ 8 MÔ HÌNH")
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

    # 2. Định nghĩa danh sách 8 mô hình
    models_dict = {
        'GCN-LSTM': {
            'class': ImprovedGNN_LSTM,
            'config': GCNLSTMConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'gcn_hidden': GCNLSTMConfig.GCN_HIDDEN,
                'lstm_hidden': GCNLSTMConfig.LSTM_HIDDEN,
                'lstm_layers': GCNLSTMConfig.LSTM_LAYERS,
                'output_feat': 1,
                'horizon': GCNLSTMConfig.HORIZON,
                'A_norm': A_norm,
                'dropout': GCNLSTMConfig.DROPOUT
            }
        },
        'Graph WaveNet': {
            'class': GraphWaveNet_Model,
            'config': WaveNetConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'residual_channels': WaveNetConfig.RESIDUAL_CHANNELS,
                'skip_channels': WaveNetConfig.SKIP_CHANNELS,
                'dilation_list': WaveNetConfig.DILATION_LIST,
                'adaptive_emb_dim': WaveNetConfig.ADAPTIVE_EMB,
                'horizon': WaveNetConfig.HORIZON,
                'output_feat': 1,
                'A_norm': A_norm,
                'dropout': WaveNetConfig.DROPOUT
            }
        },
        'GAT-TCN': {
            'class': GAT_TCN_Model,
            'config': GATConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'gat_hidden': GATConfig.GAT_HIDDEN,
                'gat_heads': GATConfig.GAT_HEADS,
                'tcn_channels': GATConfig.TCN_CHANNELS,
                'tcn_kernel': GATConfig.TCN_KERNEL,
                'horizon': GATConfig.HORIZON,
                'output_feat': 1,
                'A_raw': A_raw,
                'dropout': GATConfig.DROPOUT
            }
        },
        'ASTGCN': {
            'class': ASTGCN_Model,
            'config': ASTGCNConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': ASTGCNConfig.BLOCK_HIDDEN,
                'num_blocks': ASTGCNConfig.NUM_BLOCKS,
                'T_in': ASTGCNConfig.T_IN,
                'cheb_K': ASTGCNConfig.CHEB_K,
                'horizon': ASTGCNConfig.HORIZON,
                'output_feat': 1,
                'L_tilde': L_tilde,
                'dropout': ASTGCNConfig.DROPOUT
            }
        },
        'STGCN': {
            'class': STGCN_Model,
            'config': STGCNConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'block_hidden': STGCNConfig.BLOCK_HIDDEN,
                'num_blocks': STGCNConfig.NUM_BLOCKS,
                'T_in': STGCNConfig.T_IN,
                'cheb_K': STGCNConfig.CHEB_K,
                'horizon': STGCNConfig.HORIZON,
                'output_feat': 1,
                'L_tilde': L_tilde,
                'dropout': STGCNConfig.DROPOUT
            }
        },
        'DCRNN': {
            'class': DCRNN_Model,
            'config': DCRNNConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'hidden_dim': DCRNNConfig.HIDDEN_DIM,
                'K': DCRNNConfig.K,
                'horizon': DCRNNConfig.HORIZON,
                'output_feat': 1,
                'A_raw': A_raw,
                'dropout': DCRNNConfig.DROPOUT
            }
        },
        'AGCRN': {
            'class': AGCRN_Model,
            'config': AGCRNConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'hidden_dim': AGCRNConfig.HIDDEN_DIM,
                'embed_dim': AGCRNConfig.EMBED_DIM,
                'horizon': AGCRNConfig.HORIZON,
                'output_feat': 1,
                'dropout': AGCRNConfig.DROPOUT
            }
        },
        'TGCN': {
            'class': TGCN_Model,
            'config': TGCNConfig,
            'args': {
                'num_nodes': len(nodes),
                'in_feat': 4,
                'hidden_dim': TGCNConfig.HIDDEN_DIM,
                'horizon': TGCNConfig.HORIZON,
                'output_feat': 1,
                'A_norm': A_norm,
                'dropout': TGCNConfig.DROPOUT
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
    report_path = os.path.join(os.path.dirname(cfg.SAVE_DIR), "comparison_report.md")
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
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
