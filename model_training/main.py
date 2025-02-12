import os
from web_app.CNN import DigitCNN  # <-- Your DigitCNN class is here (with forward method)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# ---------------------------
# 1. TRAIN & EVALUATION LOGIC
# ---------------------------
def train(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch on the given data_loader.
    Returns the average training loss.
    """
    model.train()  # Put model in training mode
    running_loss = 0.0

    for images, labels in data_loader:
        # Move data to GPU/CPU device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients from previous iteration

        # ------------------------------
        # This line calls forward(images)
        # ------------------------------
        outputs = model(images)  # Forward pass

        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on the test data_loader.
    Returns (average_loss, accuracy).
    """
    model.eval()  # Put model in evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    # No gradient calculation for evaluation
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # ------------------------------
            # Again, calls forward(images)
            # ------------------------------
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------
# 2. MAIN SCRIPT
# ---------------
def main():
    # Detect if we have a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations: grayscale, resize, tensor, normalize
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # ensure 1 channel
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # mean=0.5, std=0.5
        ]
    )

    # 1) Create dataset from folder (TRAIN)
    train_dataset_path = "dataset/augmented_images/digits"  # Path to training data
    train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
    print(f"Loaded {len(train_dataset)} training images from {train_dataset_path}.")

    # 2) Load Testing Data
    test_dataset_path = "dataset/tests/test/digits"  # Path to test data
    test_dataset = ImageFolder(root=test_dataset_path, transform=transform)
    print(f"Loaded {len(test_dataset)} test images from {test_dataset_path}.")

    # 3) Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4) Initialize Model, Loss, Optimizer
    model = DigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5) Training Loop
    epochs = 5
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}]: "
            f"Train Loss = {train_loss:.4f}, "
            f"Test Loss = {test_loss:.4f}, "
            f"Test Accuracy = {test_acc:.2f}%"
        )

    # 6) Save the Trained Model
    model_save_path = "digit_cnn.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
