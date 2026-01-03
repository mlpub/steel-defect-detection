from fastapi import FastAPI
import numpy as np
#import pandas as pd
import io
import base64
#import uvicorn
from PIL import Image
from pydantic import BaseModel
#from torchvision import transforms
import onnxruntime as ort
#import config


# define constant value
IMAGE_SIZE = 384
BEST_THRESHOLD = 0.51
ONNX_MODEL = 'stage1-E1-FINAL-model.onnx'


# use Pydantic models to define the request and response schemas
# to make sure the input data is structured correctly
# and the response data is formatted properly
class PredictRequest(BaseModel):
    image: str

class PredictResponse(BaseModel):
    defect_prob: float
    defect_status: int


# resize image, convert to tensor and normalize
#inference_transform = transforms.Compose([
#    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
#    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225])
#])


# load ONNX model
ort_session = ort.InferenceSession(ONNX_MODEL, providers=['CPUExecutionProvider'])


def predict_image_onnx(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Resize image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    # Convert to numpy array and scale to [0, 1]
    img = np.array(img).astype(np.float32) / 255.0
    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # Change shape from (H, W, C) to (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    logits = ort_outs[0][0][0]
    prob = 1 / (1 + np.exp(-logits))
    pred = 1 if prob > BEST_THRESHOLD else 0
    return pred, prob


app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello World from FastAPI on Cloud Run"}


@app.post("/predict")
def predict(input_data: PredictRequest) -> PredictResponse:
    img_b64 = input_data.image

    pred1, prob1 = predict_image_onnx(img_b64)
    print(f'Predicted label: {pred1}, Probability: {prob1:.4f}')

    return PredictResponse(
        defect_prob=prob1,
        defect_status=pred1
    )
