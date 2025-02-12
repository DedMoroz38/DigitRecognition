import io
import json
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64


from CNN import DigitCNN


def predict_digit(base64_image, model_path="digit_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), 
        ]
    )

    model = DigitCNN(num_classes=10) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  

    header, encoded = base64_image.split(",", 1)
    image_bytes = base64.b64decode(encoded)

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    input_tensor = transform(image)


    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = nn.functional.softmax(
            outputs, dim=1
        )

        predicted_class = torch.argmax(probabilities, dim=1).item()

        confidence = probabilities[0, predicted_class].item() 

    return predicted_class, confidence


def main():
    try:
        data = sys.stdin.read()
        data_json = json.loads(data)
        base64_img = data_json["imageBase64"]

        digit, conf = predict_digit(base64_img)

        result = {"digit": digit, "confidence": f"{conf * 100:.2f} %"}
        print(json.dumps(result))

    except Exception as e:
        error = {"error": str(e)}
        print(json.dumps(error))


if __name__ == "__main__":
    main()
