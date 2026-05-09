# rminer - GPU Miner for rpow2

Fast GPU miner for [rpow2.com](https://rpow2.com) using Vulkan compute shaders.

## Quick Start

### Install

```bash
git clone https://github.com/growab/rminer.git
cd rminer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Get Session Cookie

1. Sign in to [rpow2.com](https://rpow2.com)
2. Open DevTools (F12) → Network tab
3. Click MINE button
4. Find `POST /challenge` request → Headers → copy `cookie:` value

```bash
export RPOW_COOKIE='rpow_session=your_cookie_here'
```

### Run

```bash
# Mine continuously
python rminer.py

# Mine 100 tokens and stop
python rminer.py --rounds 100

# Quiet mode
python rminer.py --rounds 100 --quiet
```

## Google Colab

```python
!git clone https://github.com/growab/rminer.git
%cd rminer
!pip install taichi

# Set CUDA for GPU support
import os
os.environ['TI_ARCH'] = 'cuda'

# Run miner
!python rminer.py
```

## Requirements

- Python 3.9+
- Vulkan-capable GPU (AMD/NVIDIA/Intel)
- Taichi library

## License

MIT

Author: growab
