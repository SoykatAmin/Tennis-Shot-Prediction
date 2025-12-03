# Tennis Shot Prediction

A deep learning project for predicting tennis shot locations using transformer-based neural networks. This project analyzes tennis match data to predict where players will hit their next shot based on rally context, player characteristics, and match conditions.

## 📊 Project Overview

This project implements state-of-the-art transformer models to predict tennis shot placement with the following features:

- **Player-aware modeling**: Incorporates player embeddings for personalized predictions
- **Context-sensitive predictions**: Considers court surface, score, handedness, and serve type
- **Data augmentation**: Left/right mirroring for improved generalization
- **Focal loss**: Handles class imbalance in shot zone distribution
- **Comprehensive evaluation**: Multi-metric analysis including tactical intelligence

## 🏗️ Project Structure

```
Tennis-Shot-Prediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes for loading tennis data
│   │   └── utils.py            # Data utilities and metrics
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py           # Neural network models
│   ├── utils/
│   │   └── __init__.py         # General utilities
│   └── __init__.py
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── predict.py              # Inference script
├── config/
│   ├── default_config.yaml     # Default configuration
│   ├── atp_config.yaml         # ATP-specific settings
│   └── wta_config.yaml         # WTA-specific settings
├── notebooks/
│   └── fds-tennis.ipynb        # Original research notebook
├── checkpoints/                # Model checkpoints
├── tests/                      # Unit tests
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tennis-shot-prediction.git
cd tennis-shot-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### Data Requirements

The project requires the following CSV files:
- `charting-m-points.csv` (ATP men's points data)
- `charting-m-matches.csv` (ATP men's matches data)
- `charting-w-points.csv` (WTA women's points data)  
- `charting-w-matches.csv` (WTA women's matches data)
- `atp_players.csv` (ATP player information)
- `wta_players.csv` (WTA player information)

These can be obtained from the [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject).

### Training a Model

```bash
python scripts/train.py \
    --points_path /path/to/charting-m-points.csv \
    --matches_path /path/to/charting-m-matches.csv \
    --atp_path /path/to/atp_players.csv \
    --wta_path /path/to/wta_players.csv \
    --model_type player_aware \
    --epochs 20 \
    --batch_size 64 \
    --output_dir ./checkpoints/my_model
```

### Evaluating a Model

```bash
python scripts/evaluate.py \
    --checkpoint_path ./checkpoints/my_model/best_model.pth \
    --config_path ./checkpoints/my_model/args.json \
    --points_path /path/to/charting-m-points.csv \
    --matches_path /path/to/charting-m-matches.csv \
    --atp_path /path/to/atp_players.csv \
    --wta_path /path/to/wta_players.csv \
    --output_dir ./evaluation \
    --save_plots
```

### Making Predictions

```bash
# Run demo
python scripts/predict.py \
    --model_path ./checkpoints/my_model/best_model.pth \
    --config_path ./checkpoints/my_model/args.json \
    --points_path /path/to/charting-m-points.csv \
    --matches_path /path/to/charting-m-matches.csv \
    --atp_path /path/to/atp_players.csv \
    --wta_path /path/to/wta_players.csv \
    --demo

# Single prediction
python scripts/predict.py \
    --model_path ./checkpoints/my_model/best_model.pth \
    --config_path ./checkpoints/my_model/args.json \
    --points_path /path/to/charting-m-points.csv \
    --matches_path /path/to/charting-m-matches.csv \
    --atp_path /path/to/atp_players.csv \
    --wta_path /path/to/wta_players.csv \
    --rally "4s 1f 3b" \
    --surface Clay \
    --score "30-15"
```

## 🧠 Model Architecture

### Player-Aware Transformer

The main model (`SymbolicTinyRM_PlayerAware`) combines:

- **Shot embeddings**: Zone arrival, shot type, and target embeddings
- **Player embeddings**: Unique representations for server and receiver
- **Context fusion**: Match conditions (surface, score, handedness, serve type)
- **Causal attention**: Prevents future information leakage
- **Multi-cycle processing**: Iterative refinement through transformer layers

### Context-Only Model

A simplified version (`SymbolicTinyRM_Context`) without player-specific information, suitable for scenarios with limited player data.

## 📊 Performance Metrics

The models are evaluated using multiple metrics:

- **Exact accuracy**: Percentage of perfectly predicted shots
- **Top-k accuracy**: Accuracy considering top-k predictions
- **Directional accuracy**: Left/Center/Right directional correctness
- **Winner detection**: Ability to identify winning shots (zones 7, 8, 9)
- **Tactical analysis**: Per-zone performance breakdown

### Typical Results

**Player-aware model on ATP data (20 epochs):**
- Accuracy: ~49%
- Top-3 Accuracy: ~75%
- Directional Accuracy: ~65%
- Winner Detection: ~85%

## 🔧 Configuration

The project uses YAML configuration files for easy parameter management:

```yaml
# config/atp_config.yaml
model:
  type: "player_aware"
  embed_dim: 64
  n_head: 4
  n_cycles: 3

training:
  batch_size: 64
  epochs: 20
  learning_rate: 1e-3
  focal_gamma: 2.0
```

## 🧪 Experimental Features

### Data Augmentation

The dataset automatically applies left/right mirroring to double the training data while preserving tactical validity.

### Focal Loss

Addresses class imbalance by focusing on hard-to-predict shots while down-weighting easy predictions.

### Class Weighting

Smoothed inverse frequency weighting to handle rare shot locations (e.g., winners at the net).

## 📈 Usage Examples

### Python API

```python
from scripts.predict import TennisShotPredictor

# Initialize predictor
predictor = TennisShotPredictor(
    model_path="./checkpoints/best_model.pth",
    config_path="./checkpoints/args.json",
    dataset_paths={
        'points': '/path/to/points.csv',
        'matches': '/path/to/matches.csv',
        'atp': '/path/to/atp_players.csv',
        'wta': '/path/to/wta_players.csv'
    }
)

# Make prediction
result = predictor.predict_next_shot(
    rally_string="4s 1f 3b",
    surface="Clay",
    score="30-15",
    server_hand="R",
    receiver_hand="L"
)

print(f"Top prediction: Zone {result['predictions'][0]['zone']} "
      f"({result['predictions'][0]['confidence']:.1f}%)")
```

### Jupyter Notebook

The original research and development is available in `notebooks/fds-tennis.ipynb` with detailed analysis and visualizations.

## 🔬 Research Background

This project is based on research in tennis analytics and sports prediction. Key innovations include:

1. **Temporal modeling**: Using transformer attention to capture rally dynamics
2. **Multi-modal context**: Integrating match conditions with shot sequences  
3. **Player representation**: Learning individual playing style embeddings
4. **Tactical understanding**: Evaluating beyond accuracy to tactical intelligence

## 🤝 Contributing

Contributions are welcome! Please see the following areas for improvement:

- [ ] Additional court surfaces and conditions
- [ ] Real-time prediction API
- [ ] Enhanced player modeling
- [ ] Shot quality assessment
- [ ] Visualization improvements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) for providing the tennis data
- Tennis analytics community for insights and inspiration
- PyTorch team for the deep learning framework

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{tennis_shot_prediction,
  title={Tennis Shot Prediction with Transformer Networks},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/tennis-shot-prediction}
}
```

## 🐛 Issues and Support

For bugs, feature requests, or questions:
1. Check existing [issues](https://github.com/yourusername/tennis-shot-prediction/issues)
2. Create a new issue with detailed description
3. Include code snippets and error messages when applicable

---

**Made with ❤️ for tennis analytics and deep learning**