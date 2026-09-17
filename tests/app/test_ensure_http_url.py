from types import ModuleType

import app_img
import app_vid
import pytest

MODULES = [pytest.param(app_img, id="app_img"), pytest.param(app_vid, id="app_vid")]


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/org/repo/resolve/main/model.safetensors",
        "http://example.com/lora.safetensors",
    ],
)
def test_accepts_http_urls(module: ModuleType, url: str) -> None:
    module.ensure_http_url(url)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/lora.safetensors",
        "models/lora.safetensors",
        "",
    ],
)
def test_rejects_non_http_urls(module: ModuleType, url: str) -> None:
    with pytest.raises(ValueError, match=r"Only http\(s\) URLs are supported"):
        module.ensure_http_url(url)
