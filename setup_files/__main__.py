from __future__ import annotations

from setup import stub
from txt2img import StableDiffusion


def new_stable_diffusion() -> StableDiffusion:
    return StableDiffusion()


@stub.function(gpu="A10G")
def main():
    sd = new_stable_diffusion()
    print(isinstance(sd, StableDiffusion))


if __name__ == "__main__":
    main()
