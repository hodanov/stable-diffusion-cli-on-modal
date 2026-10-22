import importlib

import pytest


@pytest.mark.parametrize("module_name", ["app_img", "app_vid"])
def test_app_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", ["app_img", "app_vid"])
def test_app_default_image_is_dockerfile_image(module_name: str) -> None:
    module = importlib.import_module(module_name)
    # modal 1.6 silently dropped the App.image setter; guard against falling back to debian_slim.
    assert module.app._get_default_image() is module.app_image  # noqa: SLF001
