import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from web_app.CNN import DigitCNN  # Import your CNN model


def extract_feature_maps(model, image):
    """
    Passes an image through the model layer by layer,
    capturing intermediate feature maps.
    """
    feature_maps = {}

    # First convolutional block
    conv1_out = model.conv1(image)  # (1, 32, 28, 28)
    conv1_out = F.relu(conv1_out)
    feature_maps["conv1"] = conv1_out

    pool1_out = model.pool(conv1_out)  # (1, 32, 14, 14)
    feature_maps["pool1"] = pool1_out

    # Second convolutional block
    conv2_out = model.conv2(pool1_out)  # (1, 64, 14, 14)
    conv2_out = F.relu(conv2_out)
    feature_maps["conv2"] = conv2_out

    pool2_out = model.pool(conv2_out)  # (1, 64, 7, 7)
    feature_maps["pool2"] = pool2_out

    return feature_maps


def visualize_feature_maps(feature_maps, layer_name, num_maps=8):
    """
    Plots feature maps from a specific layer.
    """
    feature_tensor = feature_maps[layer_name].detach().cpu().numpy()
    feature_tensor = feature_tensor[
        0
    ]  # Remove batch dimension (1, C, H, W) -> (C, H, W)

    plt.figure(figsize=(15, 5))
    for i in range(min(num_maps, feature_tensor.shape[0])):  # Limit to first 8 maps
        ax = plt.subplot(1, num_maps, i + 1)
        plt.imshow(feature_tensor[i], cmap="gray")
        plt.axis("off")
        ax.set_title(f"{layer_name} - Map {i}")

    plt.show()


def main():
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("digit_cnn.pth", map_location=device))
    model.eval()  # Put the model in evaluation mode

    # Load the single test image from manual_tests/
    image_path = "manual_tests/digit.png"  # Ensure this file exists
    image = Image.open(image_path).convert("L")  # Convert to grayscale

    # Apply the same transformations as training
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    input_tensor = (
        transform(image).unsqueeze(0).to(device)
    )  # Add batch dimension (1,1,28,28)

    # Extract feature maps
    feature_maps = extract_feature_maps(model, input_tensor)

    # Visualize feature maps from different layers
    visualize_feature_maps(feature_maps, "conv1", num_maps=16)  # First conv layer
    visualize_feature_maps(feature_maps, "pool1", num_maps=16)  # First pooling
    visualize_feature_maps(feature_maps, "conv2", num_maps=16)  # Second conv layer
    visualize_feature_maps(feature_maps, "pool2", num_maps=16)  # Second pooling


if __name__ == "__main__":
    main()
