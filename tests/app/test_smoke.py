import importlib

import pytest


@pytest.mark.parametrize("module_name", ["app_img", "app_vid"])
def test_app_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name)
