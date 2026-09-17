import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["domain", "infrastructure", "txt2img_handler", "ti2v_handler"],
)
def test_cli_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name)
