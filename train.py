import torch
import torch.nn.functional as F
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from src.models import TemporalGNN
from src.utils import load_and_preprocess_data
import argparse
import os

def train_model(model, train_dataset, test_dataset, epochs=200, learning_rate=0.01):
    """
    Train the temporal GNN model.
    
    Args:
        model: The TemporalGNN model
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: MSE: {cost.item():.4f}")
    
    # Evaluation
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    
    cost = cost / (time + 1)
    cost = cost.item()
    print(f"Test MSE: {cost:.4f}")
    
    return model, cost

def main():
    parser = argparse.ArgumentParser(description='Train SCM GNN Forecasting Model')
    parser.add_argument('--edges_path', type=str, required=True, help='Path to edges CSV file')
    parser.add_argument('--node_features_path', type=str, required=True, help='Path to node features CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--v_node_weight', type=float, default=0.1, help='Virtual node weight')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--model_save_path', type=str, default='model.pth', help='Path to save trained model')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    edges_index, edges_f, node_f, node_p = load_and_preprocess_data(
        args.edges_path, args.node_features_path, args.output_path, args.v_node_weight
    )
    
    # Create dataset
    dataset = StaticGraphTemporalSignal(
        edge_index=edges_index, 
        edge_weight=edges_f, 
        features=node_f, 
        targets=node_p
    )
    
    # Split dataset
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=args.train_ratio)
    
    # Initialize model
    node_count = node_f.shape[1]
    input_dim = node_f.shape[2]
    output_dim = 1
    
    model = TemporalGNN(
        node_count=node_count,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim
    )
    
    print(f"Model initialized with {node_count} nodes, {input_dim} input features")
    print(f"Training for {args.epochs} epochs with learning rate {args.learning_rate}")
    
    # Train model
    trained_model, test_mse = train_model(
        model, train_dataset, test_dataset, 
        epochs=args.epochs, learning_rate=args.learning_rate
    )
    
    # Save model
    torch.save(trained_model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == "__main__":
    main()

