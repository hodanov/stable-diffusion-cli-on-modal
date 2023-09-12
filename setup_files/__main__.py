from __future__ import annotations

from setup import stub
from txt2img import new_stable_diffusion


@stub.function(gpu="A10G")
def main():
    sd = new_stable_diffusion()
    print(f"Deploy '{sd.__class__.__name__}'.")


if __name__ == "__main__":
    main()
