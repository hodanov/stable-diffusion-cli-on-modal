FROM python:3.13.2-slim-bookworm
COPY ./requirements.txt /
RUN apt-get update \
    && apt-get install wget libgl1-mesa-glx libglib2.0-0 --no-install-recommends -y \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126 --no-cache-dir