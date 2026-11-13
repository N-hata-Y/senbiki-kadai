#!/usr/bin/env bash
# 黄色マーカーを基準に page_*.png を original_1.png へ位置合わせする
# 使い方: scripts/run_align_markers.sh [設定ファイル]
set -euo pipefail

# プロジェクトルート (このスクリプトの置き場所 scripts/ の1つ上)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${PROJECT_ROOT}/config/align.toml}"

uv run --project "${PROJECT_ROOT}" python "${PROJECT_ROOT}/src/senbiki_kadai/align_markers.py" \
    --config "${CONFIG}" \
    --root "${PROJECT_ROOT}"
