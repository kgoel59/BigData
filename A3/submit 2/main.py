import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score

from classifiers import NN1, NN2


def load_and_preprocess_data(train_csv: str, test_csv_1: str, test_csv_v2: str):
    try:
        # Load data
        df_train = pd.read_csv(train_csv).iloc[:, 1:]  # Skip first column
        df_test_1 = pd.read_csv(test_csv_1).iloc[:, 1:]
        df_test_v2 = pd.read_csv(test_csv_v2).iloc[:, 1:]

        # Split features and labels
        X_train = df_train.drop(['path', 'label'], axis=1)
        y_train = df_train['label']
        X_test_1 = df_test_1.drop(['path', 'label'], axis=1)
        y_test_1 = df_test_1['label']
        X_test_v2 = df_test_v2.drop(['path', 'label'], axis=1)
        y_test_v2 = df_test_v2['label']

        return X_train, y_train, X_test_1, y_test_1, X_test_v2, y_test_v2

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def train_and_evaluate(model: nn.Module, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor,
                       X_test_1_tensor: torch.Tensor, y_test_1_tensor: torch.Tensor,
                       X_test_v2_tensor: torch.Tensor, y_test_v2_tensor: torch.Tensor,
                       num_epochs: int = 10, batch_size: int = 32) -> None:
    # Define optimizer
    optimizer = Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    # Move model to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_test_1_tensor, y_test_1_tensor = X_test_1_tensor.to(device), y_test_1_tensor.to(device)
    X_test_v2_tensor, y_test_v2_tensor = X_test_v2_tensor.to(device), y_test_v2_tensor.to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()  # Set model to training mode

        total_loss, total_accuracy, total_precision = 0.0, 0.0, 0.0
        total_batches = 0

        for i in range(0, len(X_train_tensor), batch_size):
            # Batch processing
            X_batch = X_train_tensor[i:i + batch_size]
            y_batch = y_train_tensor[i:i + batch_size]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            loss = nn.CrossEntropyLoss()(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_accuracy += accuracy_score(y_batch.cpu(), predictions.cpu())
            total_precision += precision_score(y_batch.cpu(), predictions.cpu(), average='weighted', zero_division=0)
            total_batches += 1

        # Log training metrics
        train_loss = total_loss / total_batches
        train_accuracy = total_accuracy / total_batches
        train_precision = total_precision / total_batches
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Train Precision: {train_precision:.2f}')

        # Evaluation on test sets
        model.eval()  # Set model to evaluation mode
        evaluate_model(model, X_test_1_tensor, y_test_1_tensor, "Test Set 1")
        evaluate_model(model, X_test_v2_tensor, y_test_v2_tensor, "Test Set V2")

    print("Training and evaluation completed.")


def evaluate_model(model: nn.Module, X_test_tensor: torch.Tensor, y_test_tensor: torch.Tensor, dataset_name: str) -> None:
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = nn.CrossEntropyLoss()(test_outputs, y_test_tensor).item()
        test_predictions = test_outputs.argmax(dim=1)
        test_accuracy = accuracy_score(y_test_tensor.cpu(), test_predictions.cpu())
        test_precision = precision_score(y_test_tensor.cpu(), test_predictions.cpu(), average='weighted', zero_division=0)
        
        print(f'{dataset_name} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}, Precision: {test_precision:.2f}')


if __name__ == "__main__":
    # Paths to CSV files
    csv_train = '/content/finetune/pytorch-image-models-main/train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    csv_test_1 = '/content/finetune/pytorch-image-models-main/val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'
    csv_test_v2 = '/content/finetune/pytorch-image-models-main/v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv'

    # Load and preprocess data
    X_train, y_train, X_test_1, y_test_1, X_test_v2, y_test_v2 = load_and_preprocess_data(csv_train, csv_test_1, csv_test_v2)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_1_tensor = torch.tensor(X_test_1.values, dtype=torch.float32)
    y_test_1_tensor = torch.tensor(y_test_1.values, dtype=torch.long)
    X_test_v2_tensor = torch.tensor(X_test_v2.values, dtype=torch.float32)
    y_test_v2_tensor = torch.tensor(y_test_v2.values, dtype=torch.long)

    # Initialize the model
    input_size = 1024
    num_classes = 1000
    model1 = NN1(input_size, num_classes)
    model2 = NN2(input_size, num_classes)

    # Train and evaluate models
    train_and_evaluate(model1, X_train_tensor, y_train_tensor, X_test_1_tensor, y_test_1_tensor, X_test_v2_tensor, y_test_v2_tensor)
    train_and_evaluate(model2, X_train_tensor, y_train_tensor, X_test_1_tensor, y_test_1_tensor, X_test_v2_tensor, y_test_v2_tensor)
