#!/usr/bin/env bash
set -euo pipefail

# Run from the folder containing this script.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Use the Python environment activated in your terminal.
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "PS8_modularcode.ipynb" ]]; then
    echo "Missing PS8_modularcode.ipynb in this folder." >&2
    exit 1
fi

command -v "$PYTHON_BIN" >/dev/null || {
    echo "Activate your Python environment before running this script." >&2
    exit 1
}
command -v latexmk >/dev/null || {
    echo "latexmk is required to compile the LaTeX document." >&2
    exit 1
}

mkdir -p results

echo "Running PS8_modularcode.ipynb..."
"$PYTHON_BIN" -m jupyter nbconvert \
    --to notebook \
    --execute "PS8_modularcode.ipynb" \
    --output "PS8_modularcode_executed.ipynb" \
    --output-dir "." \
    --ExecutePreprocessor.timeout=-1

if [[ ! -f "PS8_FICHTNER.tex" ]]; then
    echo "Missing PS8_FICHTNER.tex. Include it in this folder or generate it in the notebook." >&2
    exit 1
fi

echo "Compiling PS8_FICHTNER.tex..."
latexmk -pdf -interaction=nonstopmode -halt-on-error "PS8_FICHTNER.tex"

echo "Finished: PS8_modularcode_executed.ipynb and PS8_FICHTNER.pdf"
