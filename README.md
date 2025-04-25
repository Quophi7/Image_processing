# Modality Separation Networks for Domain Adaptation

This repository contains the implementation of our novel approach to domain adaptation using modality separation networks. The project focuses on improving cross-domain object recognition through architectural depth and feature disentanglement.

## Overview

The project implements and evaluates a novel domain adaptation approach that leverages modality separation in deep neural networks. Our experiments demonstrate significant improvements in domain adaptation performance, particularly in challenging scenarios like Product-to-Real World and Art-to-Clipart adaptations.

## Key Features

- Modality separation approach for domain adaptation
- Comparative analysis of ResNet50 and ResNet101 architectures
- Comprehensive evaluation across multiple domain adaptation scenarios
- Detailed performance analysis and visualization tools

## Results

Our approach achieves:
- 92.47% peak accuracy in Product → Real World adaptation (ResNet101)
- 89.53% accuracy in Art → Clipart adaptation (ResNet101)
- Faster convergence compared to traditional approaches
- Improved stability in complex domain shifts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UniMoS.git
cd UniMoS
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
UniMoS/
├── configs/           # Configuration files
├── dataset/          # Dataset handling
├── ext_fea/          # External feature processing
├── log/              # Training logs
├── utils/            # Utility functions
├── model.py          # Model implementation
├── main.py           # Main training script
├── plot_results.py   # Visualization tools
└── requirements.txt  # Project dependencies
```

## Usage

1. Training:
```bash
python main.py --config configs/training_config.yaml
```

2. Visualization:
```bash
python plot_results.py --results_path log/results/
```

## Results Visualization

The repository includes several visualization tools for analyzing results:

- Training accuracy progression
- Domain adaptation comparison
- Convergence analysis
- Performance comparison
- Stability analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourcitation,
  title={Modality Separation Networks: Enhancing Domain Adaptation through Architectural Depth and Feature Disentanglement},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please contact [your-email@example.com]

## Acknowledgments

- Thanks to contributors and collaborators
- Acknowledgments for datasets used
- Any funding support
