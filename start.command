#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
if [[ ! -f vendor/tapnet/tapnet/torch/tapir_model.py ]]; then
  .venv/bin/python scripts/setup_tap.py
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 1024 --browser.gatherUsageStats false
