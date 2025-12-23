# Tennis Shot Prediction Repository

Given an ongoing tennis rally, we will predict the direction, depth, and type of the next shot. This could be used by amateur players as a sort of virtual coach for strategy. We employ 6 different models to achieve this.

## 🗂️ Repository Structure

```text
Tennis-Shot-Prediction/
├── src/
│   ├── data/           # Data loading and processing
│   ├── models/         # Neural network models (Transformers, LSTMs, Baselines)
│   └── utils/          # Utility functions for logging and evaluation
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Jupyter notebooks for demos and analysis
├── config/             # Configuration files
├── checkpoints/        # Saved model weights
└── docs/               # Documentation
```

## 🚀 Quick Start

### Data Requirements

The project requires data from the [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject):

* `charting-m-points-to-2009.csv` / `charting-m-points-2010s.csv` / `charting-m-points-2020s.csv`
* `charting-w-points-to-2009.csv` / `charting-w-points-2010s.csv` / `charting-w-points-2020s.csv`
* `charting-m-matches.csv` / `charting-w-matches.csv`
* `atp_players.csv` / `wta_players.csv`

### Training

Use the notebook: `notebooks/fds_project_models.ipynb`

### Evaluation

Use the notebook: `notebooks/fds_project_models.ipynb`
