""" Utility functions for the script. """
import random
import time
from datetime import date
from pathlib import Path

OUTPUT_DIRECTORY = "outputs"
DATE_TODAY = date.today().strftime("%Y-%m-%d")


def generate_seed() -> int:
    """
    Generate a random seed.
    """
    seed = random.randint(0, 4294967295)
    print(f"Generate a random seed: {seed}")

    return seed


def make_directory() -> Path:
    """
    Make a directory for saving outputs.
    """
    directory = Path(f"{OUTPUT_DIRECTORY}/{DATE_TODAY}")
    if not directory.exists():
        directory.mkdir(exist_ok=True, parents=True)
        print(f"Make a directory: {directory}")

    return directory


def save_prompts(inputs: dict):
    """
    Save prompts to a file.
    """
    prompts_filename = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    with open(
        file=f"{OUTPUT_DIRECTORY}/{DATE_TODAY}/prompts_{prompts_filename}.txt", mode="w", encoding="utf-8"
    ) as file:
        for name, value in inputs.items():
            file.write(f"{name} = {repr(value)}\n")
        print(f"Save prompts: {prompts_filename}.txt")


def count_token(p: str, n: str) -> int:
    """
    Count the number of tokens in the prompt and negative prompt.
    """
    token_count_p = len(p.split())
    token_count_n = len(n.split())
    if token_count_p >= token_count_n:
        token_count = token_count_p
    else:
        token_count = token_count_n

    max_embeddings_multiples = 1
    if token_count > 77:
        max_embeddings_multiples = token_count // 77 + 1

    print(f"token_count: {token_count}, max_embeddings_multiples: {max_embeddings_multiples}")

    return max_embeddings_multiples


def save_images(directory: Path, images: list[bytes], seed: int, i: int):
    """
    Save images to a file.
    """
    for j, image_bytes in enumerate(images):
        formatted_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        output_path = directory / f"{formatted_time}_{seed}_{i}_{j}.png"
        print(f"Saving it to {output_path}")
        with open(output_path, "wb") as file:
            file.write(image_bytes)
