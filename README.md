# *GRAPHINE* : Enhancing Spatiotemporal Supply Chain Forecasting Using Virtual Node-Augmented Graph Diffusion for Improved Fuel Efficiency

**Authors:** Abdulelah S. Alshehri, Azmine Toushik Wasi, Mahfuz Ahmed Anik, MD Shafikul Islam, and Mohamed Kamel Hadj-Kali


## Abstract
Spatiotemporal forecasting in supply chain networks demands modeling complex spatial dependencies and nonlinear temporal dynamics. Traditional models often fail to capture the heterogeneity, structural sparsity, and long-term dependencies of real-world supply chains. To address this, we propose GRAPHINE, a Virtual Node Diffusion-Convolutional Recurrent Neural Network tailored for supply chain forecasting. GRAPHINE leverages diffusion-based learning, using bidirectional random walks to model spatial relations and a recurrent encoder-decoder with scheduled sampling to capture temporal patterns. It introduces virtual nodes to aggregate global context and applies a learnable gating mechanism that allows each node to regulate global influence based on local features, preventing oversmoothing and preserving specificity. Evaluated on the SCG dataset for demand and inventory forecasting, GRAPHINE achieves reductions of 38.71\% in MSE and of 21.71\% in RMSE over state-of-the-art baselines, potentially lowering fuel use and related emissions by 5.3\%.  GRAPHINE thus establishes a new benchmark for applying advanced spatiotemporal graph neural architectures to complex supply chain problems, with strong implications for both economic efficiency and environmental sustainability.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AI4CHEMIA/GRAPHINE.git
   cd GRAPHINE
   ```
   *(Note: Replace `your-username` with the actual GitHub username or organization once the repository is hosted.)*

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, you can install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### Data Preparation
Place your processed supply chain data files into the `data/processed/` directory. Ensure the CSV files are correctly formatted as expected by the `data_preprocessing.py` module.

### Training the Model
To train the GNN model, run the `train.py` script from the root directory of the project. You can specify data paths and training parameters via command-line arguments or modify the `config.yaml` file.

Example command-line usage:
```bash
python train.py \
  --edges_path data/processed/Edges.csv \
  --node_features_path data/processed/NodeFeatures.csv \
  --output_path data/processed/Output.csv \
  --epochs 200 \
  --learning_rate 0.01 \
  --hidden_dim 32 \
  --v_node_weight 0.1 \
  --train_ratio 0.8 \
  --sampling_percentage 0.6 \
  --model_save_path models/trained_model.pth \
  --tag GRAPHINE_Trial
```


## Project Structure
```
GRAPHINE/
├── data/
│   ├── Edges(Plant)-Uniques.csv
│   ├── NOdeFeatures.csv
│   └── NOdeFeaturesSO.csv
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── diffconv.py
│   │   ├── dgc_rnn.py
│   │   └── temporal_gnn.py
│   └── utils/
│       ├── __init__.py
│       └── data_preprocessing.py
├── requirements.txt
├── setup.py
├── train.py
└── README.md
```

- `data/`: Directory to store raw and processed data files.
- `src/`: Contains the core Python modules of the project.
  - `src/models/`: Defines the GNN model architectures (DiffConv, DGC_RNN, TemporalGNN).
  - `src/utils/`: Provides utility functions, such as data preprocessing.
- `config.yaml`: Configuration file for project parameters.
- `requirements.txt`: Lists all Python dependencies.
- `setup.py`: Setup script for packaging the project.
- `train.py`: Main script for training the GNN model.
- `README.md`: This documentation file.


## Citation
```
@article{GRAPHINE2025,
  author  = {Alshehri, Abdulelah S. and Wasi, Azmine Toushik and Anik, Mahfuz Ahmed and Islam, MD Shafikul and Hadj-Kali, Mohamed Kamel},
  title   = {GRAPHINE: Enhancing Spatiotemporal Supply Chain Forecasting Using Virtual Node-Augmented Graph Diffusion for Improved Fuel Efficiency},
  journal = {SUBMITTED to International Journal of Production Research},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {},
  url     = {https://github.com/AI4CHEMIA/GRAPHINE},
}

```
