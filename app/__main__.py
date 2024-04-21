from __future__ import annotations

import stable_diffusion_1_5
import stable_diffusion_xl
from setup import app


@app.function(gpu="A10G")
def main():
    stable_diffusion_1_5.SD15
    stable_diffusion_xl.SDXLTxt2Img


if __name__ == "__main__":
    main.local()
