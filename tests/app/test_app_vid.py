from __future__ import annotations

from typing import TYPE_CHECKING, Any

import app_vid
import pytest
from app_vid import (
    DEFAULT_WAN_I2V_REPO_ID,
    FLOW_SHIFT_720P_AREA_THRESHOLD,
    WanI2VSetup,
    filename_from_url,
    normalize_hf_url,
    resolve_flow_shift,
    target_size_for_image,
)

if TYPE_CHECKING:
    from pathlib import Path

HF_BLOB_URL = "https://huggingface.co/org/repo/blob/main/model.safetensors"
HF_RESOLVE_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"


class TestNormalizeHfUrl:
    def test_blob_url_becomes_resolve_url(self) -> None:
        assert normalize_hf_url(HF_BLOB_URL) == HF_RESOLVE_URL

    @pytest.mark.parametrize(
        "url",
        [
            HF_RESOLVE_URL,
            # /blob/ is only rewritten on huggingface.co.
            "https://example.com/org/repo/blob/main/model.safetensors",
            "https://github.com/o/r/releases/download/v1/weights.pth",
        ],
    )
    def test_other_urls_are_untouched(self, url: str) -> None:
        assert normalize_hf_url(url) == url


class TestFilenameFromUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (HF_RESOLVE_URL, "model.safetensors"),
            (
                "https://example.com/a/b/lora.safetensors?download=true",
                "lora.safetensors",
            ),
            ("https://example.com/weights.pth#fragment", "weights.pth"),
        ],
    )
    def test_takes_the_basename_of_the_path(self, url: str, expected: str) -> None:
        assert filename_from_url(url) == expected


class TestResolveFlowShift:
    def test_override_wins(self) -> None:
        assert resolve_flow_shift(height=480, width=832, override=1.0) == 1.0

    @pytest.mark.parametrize(
        ("height", "width", "expected"),
        [
            (480, 832, 3.0),
            (720, 1280, 5.0),
            (1, FLOW_SHIFT_720P_AREA_THRESHOLD, 5.0),
            (1, FLOW_SHIFT_720P_AREA_THRESHOLD - 1, 3.0),
        ],
    )
    def test_falls_back_to_the_area_threshold(
        self, height: int, width: int, expected: float
    ) -> None:
        assert resolve_flow_shift(height=height, width=width, override=None) == expected


class TestTargetSizeForImage:
    def test_requested_size_is_kept_when_aspect_is_not_used(self) -> None:
        size = target_size_for_image(
            image_size=(1000, 500),
            height=704,
            width=1280,
            use_image_aspect=False,
            mod_value=16,
        )

        assert size == (704, 1280)

    def test_keeps_the_image_aspect_ratio(self) -> None:
        height, width = target_size_for_image(
            image_size=(1000, 500),
            height=704,
            width=1280,
            use_image_aspect=True,
            mod_value=16,
        )

        # A 2:1 landscape image stays roughly 2:1 ...
        assert width / height == pytest.approx(2.0, abs=0.05)
        # ... and both sides are multiples of mod_value.
        assert height % 16 == 0
        assert width % 16 == 0

    def test_area_is_capped_by_the_requested_size(self) -> None:
        height, width = target_size_for_image(
            image_size=(400, 400),
            height=256,
            width=256,
            use_image_aspect=True,
            mod_value=16,
        )

        assert height * width <= 256 * 256

    def test_sides_never_fall_below_mod_value(self) -> None:
        height, width = target_size_for_image(
            image_size=(4000, 4),
            height=64,
            width=64,
            use_image_aspect=True,
            mod_value=16,
        )

        assert min(height, width) == 16


def make_config(**model_overrides: object) -> dict[str, Any]:
    model: dict[str, Any] = {"name": "wan22"}
    model.update(model_overrides)
    return {"wan_i2v": {"model": model}}


class TestWanI2VSetupValidation:
    @pytest.mark.parametrize(
        ("config", "message"),
        [
            ({}, "wan_i2v is required"),
            ({"wan_i2v": {}}, "wan_i2v.model is required"),
            (
                make_config(safetensors_url_low="https://example.com/low.safetensors"),
                "safetensors_url is required when safetensors_url_low is set",
            ),
        ],
    )
    def test_rejects_invalid_config(self, config: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            WanI2VSetup(config, "")

    def test_repo_id_falls_back_to_the_default(self, downloads: Downloads) -> None:
        WanI2VSetup(make_config(), "").download_model()

        assert downloads.snapshots[0]["repo_id"] == DEFAULT_WAN_I2V_REPO_ID

    def test_repo_id_can_be_overridden(self, downloads: Downloads) -> None:
        WanI2VSetup(make_config(repo_id="me/wan"), "").download_model()

        assert downloads.snapshots[0]["repo_id"] == "me/wan"


class Downloads:
    """Records what download_model() would have fetched."""

    def __init__(self) -> None:
        self.snapshots: list[dict[str, Any]] = []
        self.files: list[tuple[str, Path]] = []
        self.commits = 0


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> Downloads:
    recorded = Downloads()

    def fake_snapshot_download(**kwargs: Any) -> str:  # noqa: ANN401 - mirrors snapshot_download
        recorded.snapshots.append(kwargs)
        return kwargs.get("local_dir", "")

    def fake_download_file(_self: WanI2VSetup, url: str, cache_path: Path) -> None:
        recorded.files.append((url, cache_path))

    class FakeVolume:
        def commit(self) -> None:
            recorded.commits += 1

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(WanI2VSetup, "_WanI2VSetup__download_file", fake_download_file)
    monkeypatch.setattr(app_vid, "model_volume", FakeVolume())
    return recorded


class TestWanI2VSetupDownloadModel:
    def test_downloads_the_whole_repo_without_a_safetensors_url(
        self, downloads: Downloads
    ) -> None:
        expected_auth = "hf-token"

        WanI2VSetup(make_config(), expected_auth).download_model()

        assert len(downloads.snapshots) == 1
        snapshot = downloads.snapshots[0]
        assert snapshot["token"] == expected_auth
        assert snapshot["local_dir"].endswith("/wan22")
        assert snapshot["ignore_patterns"] == ["assets/*", "examples/*", "*.md"]
        assert "allow_patterns" not in snapshot
        assert downloads.files == []
        assert downloads.commits == 1

    def test_empty_token_is_passed_as_none(self, downloads: Downloads) -> None:
        WanI2VSetup(make_config(), "").download_model()

        assert downloads.snapshots[0]["token"] is None

    def test_high_noise_url_skips_the_transformer_and_fetches_transformer_2(
        self, downloads: Downloads
    ) -> None:
        url = "https://example.com/high.safetensors"

        WanI2VSetup(make_config(safetensors_url=url), "").download_model()

        assert len(downloads.snapshots) == 2
        skipped = downloads.snapshots[0]["ignore_patterns"]
        assert "transformer/*.safetensors" in skipped
        assert "transformer_2/*.safetensors" not in skipped
        # transformer_2 is still pulled from the repo in a second pass.
        assert downloads.snapshots[1]["allow_patterns"] == ["transformer_2/*"]
        assert [url for url, _ in downloads.files] == [url]
        assert downloads.files[0][1].name == "transformer"
        assert downloads.commits == 1

    def test_both_urls_skip_both_transformers(self, downloads: Downloads) -> None:
        high = "https://example.com/high.safetensors"
        low = "https://example.com/low.safetensors"

        WanI2VSetup(
            make_config(safetensors_url=high, safetensors_url_low=low),
            "",
        ).download_model()

        assert len(downloads.snapshots) == 1
        skipped = downloads.snapshots[0]["ignore_patterns"]
        assert "transformer/*.safetensors" in skipped
        assert "transformer_2/*.safetensors" in skipped
        assert [url for url, _ in downloads.files] == [high, low]
        assert [path.name for _, path in downloads.files] == [
            "transformer",
            "transformer_2",
        ]
        assert downloads.commits == 1
