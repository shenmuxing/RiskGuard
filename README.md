# Account Risk Detection in Large-Scale Financial Graphs with Auxiliary Asset Prediction

**RiskGuard**: A novel framework for detecting forced liquidation risks in leveraged trading accounts through auxiliary asset prediction and gradient-based graph sampling.

This is the source code for the paper "Account Risk Detection in Large-Scale Financial Graphs with Auxiliary Asset Prediction", addressing the critical challenge of account risk detection in large-scale financial networks by jointly modeling asset fluctuations and account behaviors in temporal bipartite graphs.

The full version of our paper is available at [full_version/RiskGuard-full.pdf](https://github.com/shenmuxing/RiskGuard/tree/main/full_version/RiskGuard-full.pdf)

## Environment Setup

### Prerequisites
- CUDA-capable GPU (NVIDIA GPU with compute capability >= 3.5)
- CUDA Toolkit >= 11.7
- Python >= 3.9

### Dependencies
```bash
# Core dependencies
numpy==1.26.4
scikit-learn>=1.5.1
torch>=2.0.0+cu117  # CUDA 11.7 version
torch-geometric>=2.5.3
```

## Training and Evaluation

The training and evaluation procedures are implemented in `main.py`. 

### Quick Start

To train and evaluate the model on the default dataset, simply run:
```bash
python main.py --hidden_dim 256
```

## License

This project is licensed under the Mozilla Public License Version 2.0 - see the [LICENSE](LICENSE) file for details.

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)