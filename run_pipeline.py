import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "src" / "volatility"

STAGE_SCRIPTS = {
    "diagnostics": ["cli.py"],
    "benchmark": ["benchmark.py"],
    "final-test": ["final_test.py"],
    "predict": ["predict.py"],
    "all": [
        "cli.py",
        "benchmark.py",
        "final_test.py",
        "predict.py",
    ],
}


def run_stage(stage: str, interactive_plots: bool = False) -> None:
    env = os.environ.copy()
    if not interactive_plots:
        env["MPLBACKEND"] = "Agg"
        env["PYTHONWARNINGS"] = (
            "ignore:FigureCanvasAgg is non-interactive"
        )
    matplotlib_cache = Path(tempfile.gettempdir()) / "quant_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(matplotlib_cache)

    for script_name in STAGE_SCRIPTS[stage]:
        script_path = CODE_DIR / script_name
        print(f"\n{'=' * 72}", flush=True)
        print(f"运行阶段：{script_name}", flush=True)
        print(f"{'=' * 72}", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=CODE_DIR,
            env=env,
            check=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行 HAR-RV 波动率预测项目的指定阶段。"
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_SCRIPTS,
        default="all",
        help="选择运行阶段；默认运行完整流程。",
    )
    parser.add_argument(
        "--interactive-plots",
        action="store_true",
        help="显示 Matplotlib 窗口；默认使用无界面模式。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_stage(
        stage=args.stage,
        interactive_plots=args.interactive_plots,
    )


if __name__ == "__main__":
    main()
