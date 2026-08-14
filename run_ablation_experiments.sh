#!/bin/bash

echo "======================================================"
echo "🚀 1. RUNNING DYNAMIC GRAPH ABLATION"
echo "======================================================"
python run_dynamic_graph_ablation.py
echo "✅ DYNAMIC GRAPH ABLATION FINISHED."
echo ""

echo "======================================================"
echo "🚀 2. RUNNING ATTENTION POSITION ABLATION"
echo "======================================================"
python run_attention_position_ablation.py
echo "✅ ATTENTION POSITION ABLATION FINISHED."
echo ""

echo "🎉 TẤT CẢ ABLATION EXPERIMENTS ĐÃ HOÀN THÀNH!"
