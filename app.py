# Author: Charles R. Clark
# CS 6440 Spring 2024

from typing import Tuple
import os

import torch
import torch.nn as nn
import torchvision.transforms
from torchvision.io import read_image

from flask import Flask, request, session
from flask_cors import CORS

from data.data_utils import MAPPING
from models import FinalModel

IMG_SIZE = 64
TMP_FOLDER = os.path.join(os.path.dirname(__file__), 'tmp')
ACCEPTED_EXTENSIONS = {'jpg', 'jpeg'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
app.config['UPLOADS_FOLDER'] = TMP_FOLDER
app.config['SECRET_KEY'] = os.environ.get('secret_key')

CORS(app, resources=r'/*')

model = FinalModel(init_img_size=IMG_SIZE)
model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'saved_models', 'best_FinalModel_weights.pth'), map_location=torch.device(DEVICE)))
model.eval()

def transform(input: torch.Tensor) -> torch.Tensor:
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize((0.5, ), (0.5))
    ])

    transformed_input = transforms(input).unsqueeze(dim=0)

    return transformed_input

def predict(model: nn.Module, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    transformed_input = transform(input)

    scores = model(transformed_input)
    probs = nn.functional.softmax(scores, dim=1)

    y_prob = torch.max(probs)
    y_pred = torch.argmax(probs)

    return (y_pred, y_prob)

@app.route('/is_image', methods=['POST'])
def is_image():
    if request.method == 'POST':
        file = request.files['file']

        file_extension = file.filename.split('.')[-1]
        if file_extension in ACCEPTED_EXTENSIONS:
            outputs = {
                'is_image': 1
            }
        else:
            outputs = {
                'is_image': 0
            }

        return outputs

@app.route('/get_prediction', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        file = request.files['file']

        dirpath = os.path.join(os.path.dirname(__file__), TMP_FOLDER)
        path = os.path.join(dirpath, file.filename)
        file.save(dst=path)

        session['uploadPath'] = path

        file_extension = file.filename.split('.')[-1]
        if file_extension in ACCEPTED_EXTENSIONS:
            input_img = read_image(path).float()
            
            y_pred, y_prob = predict(model, input_img)
            outputs = {
                'class': list(MAPPING.keys())[int(y_pred)],
                'prob': round(float(y_prob), 6) * 100
            }
        else:
            outputs = {
                'class': 'error',
                'prob': -1.0
            }

        try:
            os.remove(path=path)
        except FileNotFoundError:
            print(f'\nERROR -> COULDN\'T DELETE FILE AT PATH "{path}"\n')

        return outputs

@app.route('/', methods=['GET'])
def identity():
    return {
        'response': 'Hi, I\'m classif-MRI'
    }

if __name__ == '__main__':
    app.secret_key = os.environ.get('SECRET_KEY')
    app.run(debug=False)


