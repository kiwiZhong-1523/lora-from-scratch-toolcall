"""Download the models and the raw dataset from ModelScope into models/ and data/raw/.

    python scripts/download.py            # 0.5B + 1.5B + xLAM
    python scripts/download.py --skip-1.5b
"""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download
from modelscope.hub.file_download import dataset_file_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-1.5b", dest="skip_1_5b", action="store_true")
    args = ap.parse_args()

    models = ["Qwen/Qwen2.5-0.5B-Instruct"] + ([] if args.skip_1_5b else ["Qwen/Qwen2.5-1.5B-Instruct"])
    for m in models:
        print(snapshot_download(m, local_dir=str(ROOT / "models" / m.split("/")[1])))
    print(dataset_file_download("Salesforce/xlam-function-calling-60k", "xlam_function_calling_60k.json",
                                local_dir=str(ROOT / "data" / "raw")))


if __name__ == "__main__":
    main()
