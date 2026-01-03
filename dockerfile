FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /steel-defect-detection

RUN pip install uv
COPY pyproject.toml .
COPY uv.lock .
#COPY src/common_utils.py .
#COPY src/config.py .
#COPY src/data_utils.py .
#COPY src/plot_utils.py .
#COPY src/train_utils.py .
COPY src/predict.py .
COPY src/stage1-E1-FINAL-model.onnx .

RUN uv sync
#RUN uv pip install --target /var/task numpy fastapi pydantic pillow uvicorn
#RUN uv pip install --target /var/task torch torchvision --index-url https://download.pytorch.org/whl/cpu
#RUN uv pip install --target /var/task onnx onnxruntime

EXPOSE 9696

CMD ["uv", "run", "predict.py"]