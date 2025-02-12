import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# If your CNN class is in a separate file "CNN.py"
from web_app.CNN import DigitCNN


def predict_digit(image_path, model_path="digit_cnn.pth"):
    # 1) Choose device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Define the same transformations used during training
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # ensure 1 channel
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # must match training normalization
        ]
    )

    # 3) Load the trained model
    model = DigitCNN(num_classes=10)  # Create the same model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Evaluation mode (no dropout, etc.)

    # 4) Load the image
    # If you only have one image in manual_tests/, you can specify it directly
    # or pass the path in as 'image_path'.
    image = Image.open(image_path).convert("L")  # Convert to grayscale explicitly

    # 5) Transform the image (just like training transforms)
    input_tensor = transform(image)

    # 6) Create a batch dimension by unsqueezing
    input_tensor = input_tensor.unsqueeze(0).to(device)
    # Shape now: (1, 1, 28, 28)

    # 7) Forward pass (no gradient)
    with torch.no_grad():
        outputs = model(input_tensor)  # Shape: (1, 10)
        probabilities = nn.functional.softmax(
            outputs, dim=1
        )  # Convert logits to probabilities

        # 8) Get predicted class index
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # 9) Probability of that class
        confidence = probabilities[0, predicted_class].item()  # e.g., 0.82 for 82%

    # 10) Print or return the result
    print(f"Recognised digit: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    # Path to your saved model
    model_path = "web_app/digit_cnn.pth"

    # Suppose there's exactly one image in manual_tests/ folder
    # Example: "manual_tests/my_digit.png"
    test_image_path = os.path.join("manual_tests", "digit.png")

    predict_digit(test_image_path, model_path)
