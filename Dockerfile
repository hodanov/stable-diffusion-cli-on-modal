FROM python:3.11.3-slim-bullseye
COPY requirements.txt /
RUN apt update \
    && apt install -y wget git libgl1-mesa-glx libglib2.0-0 \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 \
    && mkdir -p /vol/cache/esrgan \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /vol/cache/esrgan \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P /vol/cache/esrgan \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P /vol/cache/esrgan \
    && wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P /vol/cache/esrgan
