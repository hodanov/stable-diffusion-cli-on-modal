from __future__ import annotations

from setup import stub
from stable_diffusion_1_5 import Txt2Img


@stub.function(gpu="A10G")
def main():
    Txt2Img


if __name__ == "__main__":
    main.local()
