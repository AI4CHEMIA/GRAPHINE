import torch 
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from src.models import TemporalGNN
from src.utils import load_and_preprocess_data
import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, explained_variance_score

def train_model(model, train_dataset, test_dataset, epochs=200, learning_rate=0.01, sampling_percentage=1.0, filename_prefix="trial"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_logs = []

    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_preds = []
        epoch_targets = []
        total_loss = 0

        for time, snapshot in enumerate(train_dataset):
            num_edges = snapshot.edge_index.shape[1]
            num_sampled_edges = int(num_edges * sampling_percentage)
            sampled_edge_indices = np.random.choice(num_edges, num_sampled_edges, replace=False)
            edge_index = snapshot.edge_index[:, sampled_edge_indices]
            edge_attr = snapshot.edge_attr[sampled_edge_indices]

            y_hat = model(snapshot.x, edge_index, edge_attr)
            loss = torch.mean((y_hat - snapshot.y) ** 2)

            total_loss += loss
            epoch_preds.append(y_hat.detach().cpu().numpy())
            epoch_targets.append(snapshot.y.detach().cpu().numpy())

        total_loss = total_loss / (time + 1)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = np.concatenate(epoch_preds).flatten()
        targets = np.concatenate(epoch_targets).flatten()
        diff = preds - targets

        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, preds)
        r2 = r2_score(targets, preds)
        me = max_error(targets, preds)
        evs = explained_variance_score(targets, preds)
        mean_error = np.mean(diff)

        train_logs.append({
            "epoch": epoch,
            "loss": total_loss.item(),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "max_error": me,
            "explained_variance": evs,
            "mean_error": mean_error
        })

    if filename_prefix:
        pd.DataFrame(train_logs).to_csv(f"{filename_prefix}.csv", index=False)

    model.eval()
    test_preds = []
    test_targets = []
    mean_diff_tensor = 0
    cost_tensor = 0

    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        diff = y_hat - snapshot.y
        cost_tensor += torch.mean(diff ** 2)
        mean_diff_tensor += torch.abs(torch.mean(diff))

        test_preds.append(y_hat.detach().cpu().numpy())
        test_targets.append(snapshot.y.detach().cpu().numpy())

    test_preds = np.concatenate(test_preds).flatten()
    test_targets = np.concatenate(test_targets).flatten()
    diff = test_preds - test_targets

    test_mse = mean_squared_error(test_targets, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    test_me = max_error(test_targets, test_preds)
    test_evs = explained_variance_score(test_targets, test_preds)
    test_mean_error = np.mean(diff)
    test_mean_diff_tensor = (mean_diff_tensor / (time + 1)).item()

    if filename_prefix:
        with open(f"{filename_prefix}.txt", "w") as f:
            f.write("Final Test Results:\n")
            f.write(f"MSE: {test_mse:.6f}\n")
            f.write(f"RMSE: {test_rmse:.6f}\n")
            f.write(f"MAE: {test_mae:.6f}\n")
            f.write(f"R² Score: {test_r2:.6f}\n")
            f.write(f"Max Error: {test_me:.6f}\n")
            f.write(f"Explained Variance: {test_evs:.6f}\n")
            f.write(f"Mean Error: {test_mean_error:.6f}\n")
            f.write(f"Mean Diff (tensor, across time): {test_mean_diff_tensor:.6f}\n")

    print("Test Scores:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    print(f"R² Score: {test_r2:.6f}")
    print(f"Max Error: {test_me:.6f}")
    print(f"Explained Variance: {test_evs:.6f}")
    print(f"Mean Error: {test_mean_error:.6f}")
    print(f"Mean Diff (tensor, across time): {test_mean_diff_tensor:.6f}")

    return model

def main():
    parser = argparse.ArgumentParser(description='Train SCM GNN Forecasting Model')
    parser.add_argument('--edges_path', type=str, required=True)
    parser.add_argument('--node_features_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--v_node_weight', type=float, default=0.1)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--sampling_percentage', type=float, default=1.0)
    parser.add_argument('--model_save_path', type=str, default='model.pth')
    parser.add_argument('--tag', type=str, default="timeseries_run")

    args = parser.parse_args()

    edges_index, edges_f, node_f, node_p = load_and_preprocess_data(
        args.edges_path, args.node_features_path, args.output_path, args.v_node_weight
    )

    dataset = StaticGraphTemporalSignal(
        edge_index=edges_index, 
        edge_weight=edges_f, 
        features=node_f, 
        targets=node_p
    )

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)

    node_count = node_f.shape[1]
    input_dim = node_f.shape[2]
    output_dim = 1

    model = TemporalGNN(
        node_count=node_count,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim
    )

    filename_prefix = f"results_VNW-{args.v_node_weight}_LR-{args.learning_rate}_SP-{int(args.sampling_percentage*100)}--{args.tag}"
    print(f"Model initialized with {node_count} nodes, {input_dim} input features")
    print(f"Training for {args.epochs} epochs with learning rate {args.learning_rate}, edge sampling {args.sampling_percentage*100:.0f}%")

    trained_model = train_model(
        model, train_dataset, test_dataset, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        sampling_percentage=args.sampling_percentage,
        filename_prefix=filename_prefix
    )

    torch.save(trained_model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()
