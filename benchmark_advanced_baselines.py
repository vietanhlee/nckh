import sys
from benchmark_5seeds import run_benchmark

if __name__ == "__main__":
    # Thêm tham số mặc định --model_group advanced
    if "--model_group" not in sys.argv:
        sys.argv.extend(["--model_group", "advanced"])
    run_benchmark()
