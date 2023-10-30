FROM python:3.11.3-slim-bullseye
COPY ./requirements.txt /
RUN apt-get update \
    && apt-get install wget libgl1-mesa-glx libglib2.0-0 --no-install-recommends -y \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir \
    && mkdir -p /vol/cache/esrgan \
    && wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /vol/cache/esrgan \
    && wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -P /vol/cache/esrgan \
    && wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P /vol/cache/esrgan \
    && wget --progress=dot:giga https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P /vol/cache/esrgan \
    && wget --progress=dot:giga https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P /vol/cache/esrgan
