from flask import Flask, request, jsonify
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import io
from torch.serialization import safe_globals
import os

app = Flask(__name__)

# Define model architecture to match training
class CancerClassifier(nn.Module):
    def __init__(self):
        super(CancerClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features
        
        # Match the classifier architecture from training
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.resnet.fc = self.classifier

    def forward(self, x):
        return self.resnet(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CancerClassifier()

try:
    # Load checkpoint with safe globals for numpy scalar
    with safe_globals(['numpy._core.multiarray.scalar']):
        checkpoint = torch.load('cancer_classifier_best.pth', weights_only=False, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

model.eval()
model.to(torch.device('cpu'))

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prob = torch.sigmoid(output).item()
            label = 1 if prob > 0.25 else 0  # Using lower threshold from training

        return jsonify({
            "prediction": "cancer" if label == 1 else "no cancer",
            "probability": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
