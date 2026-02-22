# LA Traffic GNN

GCN+LSTM traffic speed forecasting on the METR-LA dataset. The model uses Graph Convolutional Networks to capture spatial dependencies between 207 highway sensors and an LSTM to model temporal patterns, predicting traffic speeds 1 hour ahead (12 steps at 5-minute intervals) from 1 hour of history.

Based on the T-GCN architecture (Zhao et al., 2020), implemented with PyTorch and PyTorch Geometric.

## Architecture

```
Input (batch, 12, 207, 2)         [speed, time-of-day]
        |
  for t in 1..12:
        |
   2-layer GCN                    spatial aggregation over sensor graph
        |
     LSTMCell                     temporal state update
        |
  Linear decoder                  project hidden state -> 12 future steps
        |
Output (batch, 207, 12)           predicted speeds per sensor
```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd la-traffic-gnn
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Download METR-LA data
python -m src.data.download
# or: make download
```

**Note:** The raw data files (~57 MB) are hosted on Google Drive. If `gdown` fails due to rate limiting, download manually from the [DCRNN repository](https://github.com/liyaguang/DCRNN) and place files in `data/raw/`.

## Usage

```bash
# 1. Preprocess: HDF5 -> sliding windows
python -m src.data.preprocess

# 2. Train (auto-detects GPU)
python scripts/train.py --config configs/default.yaml

# 3. Evaluate on test set
python scripts/evaluate.py --config configs/default.yaml

# 4. Streaming replay (simulates real-time predictions)
python scripts/stream_replay.py --config configs/default.yaml --mode fast

# 5. Export for Tableau
python scripts/export_tableau.py --config configs/default.yaml
```

Or use the Makefile: `make preprocess`, `make train`, `make evaluate`, `make stream`, `make export`.

## Project Structure

```
la-traffic-gnn/
├── configs/
│   └── default.yaml              # all hyperparameters
├── data/
│   ├── raw/                      # METR-LA source files (.gitignored)
│   └── processed/                # preprocessed .npz splits (.gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/
│   ├── train.py                  # training entry point
│   ├── evaluate.py               # test-set evaluation + plots
│   ├── export_tableau.py         # star-schema export
│   └── stream_replay.py          # streaming simulation
├── src/
│   ├── data/
│   │   ├── download.py           # METR-LA download via gdown
│   │   ├── preprocess.py         # HDF5 -> sliding windows
│   │   ├── graph.py              # adjacency matrix construction
│   │   └── dataset.py            # PyTorch Dataset/DataLoader
│   ├── models/
│   │   └── gcn_lstm.py           # T-GCN model (~38K params)
│   ├── training/
│   │   ├── trainer.py            # training loop + early stopping
│   │   └── metrics.py            # masked MAE, RMSE, MAPE
│   ├── streaming/
│   │   ├── simulator.py          # test-set replay engine
│   │   └── writer.py             # parquet + CSV output
│   ├── export/
│   │   └── tableau.py            # star-schema generation
│   └── viz/
│       ├── graph_viz.py          # sensor network plots
│       ├── timeseries.py         # actual vs predicted overlays
│       └── training_curves.py    # loss/metric curves
├── outputs/                      # checkpoints, figures, exports (.gitignored)
├── pyproject.toml
├── Makefile
└── PROJECT_LOG.md                # development log
```

## Results

| Metric | Value |
|--------|-------|
| Val MAE (normalized) | 0.3981 |
| Val RMSE (normalized) | 0.7313 |
| Val MAE (mph) | ~3.88 |
| Best epoch | 63 |

## Dataset

**METR-LA** (Li et al., 2018): 4 months of traffic speed readings from 207 loop detectors on Los Angeles County highways, recorded at 5-minute intervals (34,272 timesteps). The sensor graph is constructed using a Gaussian kernel over road-network distances with epsilon thresholding.

## References

- Zhao, L. et al. (2020). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction. *IEEE Transactions on Intelligent Transportation Systems*.
- Li, Y. et al. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. *ICLR 2018*.
- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.
- Anthropic. (2025). Claude 4.6 Opus (February 2026) [Large language model]. https://claude.ai/

## Note From Lucas Young
I use claude code as a coding assistant during large projects. This was used during my portion of the project in a way that accelerated my learing of the material. At every step of the way I ensure that I fully understand the mechanisms behind any code of which AI helps me to write. In doing so, I am able to achieve much more in the period of time than I would otherwise be able to accomplish. My description of my AI usage does not reflect any usage of my teammates.
-Lucas Young
