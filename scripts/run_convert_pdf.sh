#!/usr/bin/env bash
# PDF を PNG 画像に変換する (src/senbiki_kadai/convert_pdf.py を実行)
set -euo pipefail

# プロジェクトルート (このスクリプトの置き場所 scripts/ の1つ上)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_PATH="${PROJECT_ROOT}/references/JPAN_original_kmu.pdf"
OUTPUT_DIR="${PROJECT_ROOT}/output/test_images"


# pdf2image は poppler (pdftoppm) が必要
if ! command -v pdftoppm >/dev/null 2>&1; then
    echo "pdftoppm が見つかりません。'brew install poppler' でインストールしてください。" >&2
    exit 1
fi

# pdf2image は pyproject.toml の依存に無いため --with で一時的に追加して実行
uv run  "${PROJECT_ROOT}/src/senbiki_kadai/convert_pdf.py" \
--pdf_path "${PDF_PATH}" \
--output_dir "${OUTPUT_DIR}"