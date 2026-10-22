from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import infrastructure
import pytest
from domain import Prompts, Seed, VideoPrompts
from infrastructure import (
    SDXLTxt2Img,
    Ti2VInterface,
    Txt2ImgInterface,
    WanTI2V,
    new_ti2v,
    new_txt2img,
    wait_for_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeClock:
    """Stands in for the time module: only monotonic(), advanced by FakeFunctionCall."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class FakeFunctionCall:
    """Stands in for modal.FunctionCall: get() times out `pending_polls` times, then returns."""

    object_id = "fc-test"

    def __init__(
        self,
        result: object,
        *,
        pending_polls: int = 0,
        clock: FakeClock | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result
        self.pending_polls = pending_polls
        self.clock = clock or FakeClock()
        self.error = error
        self.timeouts: list[float | None] = []
        self.cancelled = False

    def get(self, timeout: float | None = None) -> object:
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        if len(self.timeouts) <= self.pending_polls:
            # The real get() blocks for `timeout` seconds before giving up.
            self.clock.now += timeout or 0
            raise TimeoutError
        return self.result

    def cancel(self) -> None:
        self.cancelled = True


class FakeMethod:
    """Stands in for a Modal method: records the kwargs passed to .remote() and .spawn()."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.spawn_calls: list[dict[str, Any]] = []

    def remote(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result

    def spawn(self, **kwargs: object) -> FakeFunctionCall:
        self.spawn_calls.append(kwargs)
        return FakeFunctionCall(self.result)


class FakeCls:
    def __init__(self, result: object) -> None:
        self.run_inference = FakeMethod(result)

    def __call__(self) -> FakeCls:
        # modal.Cls instances are called to get an instance handle.
        return self


@pytest.fixture
def fake_from_name(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], FakeCls]:
    def install(result: object) -> FakeCls:
        fake = FakeCls(result)
        lookups: list[tuple[str, str]] = []

        def fake_cls_from_name(app_name: str, name: str) -> FakeCls:
            lookups.append((app_name, name))
            return fake

        monkeypatch.setattr(infrastructure.modal.Cls, "from_name", fake_cls_from_name)
        fake.lookups = lookups  # type: ignore[attr-defined]
        return fake

    return install


def make_prompts() -> Prompts:
    return Prompts(
        prompt="a cat",
        n_prompt="blurry",
        height=512,
        width=768,
        samples=2,
        steps=20,
    )


def make_video_prompts(**overrides: object) -> VideoPrompts:
    kwargs: dict[str, Any] = {
        "prompt": "a cat",
        "n_prompt": "blurry",
        "height": 512,
        "width": 768,
        "samples": 1,
        "steps": 20,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
        "use_image_aspect": True,
        "use_upscaler": False,
        "use_face_restore": False,
        "image_path": "input.png",
    }
    kwargs.update(overrides)
    return VideoPrompts(**kwargs)


class TestNewTxt2Img:
    def test_sdxl_builds_client(
        self, fake_from_name: Callable[[object], FakeCls]
    ) -> None:
        fake = fake_from_name([])

        txt2img = new_txt2img("sdxl", make_prompts(), "png", use_upscaler=False)

        assert isinstance(txt2img, SDXLTxt2Img)
        assert isinstance(txt2img, Txt2ImgInterface)
        assert fake.lookups == [("sdxl-cli", "SDXLTxt2Img")]

    def test_unknown_version_is_rejected(
        self, fake_from_name: Callable[[object], FakeCls]
    ) -> None:
        fake_from_name([])

        with pytest.raises(ValueError, match="Invalid version: sd15"):
            new_txt2img("sd15", make_prompts(), "png", use_upscaler=False)


class TestSDXLTxt2Img:
    def test_run_inference_forwards_arguments(
        self, fake_from_name: Callable[[object], FakeCls]
    ) -> None:
        fake = fake_from_name([b"image"])
        txt2img = new_txt2img("sdxl", make_prompts(), "avif", use_upscaler=True)

        images = txt2img.run_inference(Seed(42))

        assert images == [b"image"]
        # spawn() rather than remote(), so the wait can be logged and bounded.
        assert fake.run_inference.calls == []
        assert fake.run_inference.spawn_calls == [
            {
                "prompt": "a cat",
                "n_prompt": "blurry",
                "height": 512,
                "width": 768,
                "steps": 20,
                "seed": 42,
                "use_upscaler": True,
                "output_format": "avif",
            },
        ]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(infrastructure, "time", fake)
    return fake


class TestWaitForResult:
    def test_keeps_polling_until_the_result_arrives(
        self, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        call = FakeFunctionCall([b"image"], pending_polls=2, clock=clock)

        with caplog.at_level(logging.INFO):
            result = wait_for_result(call, poll_interval=30, max_wait=900)

        assert result == [b"image"]
        assert call.timeouts == [30, 30, 30]
        assert not call.cancelled
        waiting = [
            r.getMessage() for r in caplog.records if "Still waiting" in r.getMessage()
        ]
        assert len(waiting) == 2
        assert "fc-test" in waiting[0]

    def test_cancels_and_raises_after_max_wait(self, clock: FakeClock) -> None:
        call = FakeFunctionCall([b"image"], pending_polls=100, clock=clock)

        with pytest.raises(TimeoutError, match="fc-test"):
            wait_for_result(call, poll_interval=30, max_wait=90)

        assert call.cancelled
        assert call.timeouts == [30, 30, 30]

    def test_cancels_when_interrupted(self, clock: FakeClock) -> None:
        call = FakeFunctionCall([b"image"], clock=clock, error=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            wait_for_result(call, poll_interval=30, max_wait=900)

        assert call.cancelled

    def test_remote_failure_is_raised_without_cancelling(
        self, clock: FakeClock
    ) -> None:
        call = FakeFunctionCall([b"image"], clock=clock, error=RuntimeError("boom"))

        with pytest.raises(RuntimeError, match="boom"):
            wait_for_result(call, poll_interval=30, max_wait=900)

        assert not call.cancelled


class TestNewTi2V:
    def test_builds_client(self, fake_from_name: Callable[[object], FakeCls]) -> None:
        fake = fake_from_name(b"")

        ti2v = new_ti2v(
            prompts=make_video_prompts(),
            num_frames=81,
            fps=16,
            guidance_scale=5.0,
            guidance_scale_2=None,
            use_image_aspect=True,
            use_upscaler=False,
            use_face_restore=False,
        )

        assert isinstance(ti2v, WanTI2V)
        assert isinstance(ti2v, Ti2VInterface)
        assert fake.lookups == [("wan-i2v-cli", "WanTI2V")]


class TestWanTI2V:
    def test_run_inference_forwards_arguments(
        self, fake_from_name: Callable[[object], FakeCls]
    ) -> None:
        fake = fake_from_name(b"mp4")
        ti2v = new_ti2v(
            prompts=make_video_prompts(),
            num_frames=121,
            fps=24,
            guidance_scale=5.0,
            guidance_scale_2=3.5,
            use_image_aspect=False,
            use_upscaler=True,
            use_face_restore=True,
        )

        video = ti2v.run_inference(Seed(7), b"png-bytes")

        assert video == b"mp4"
        assert fake.run_inference.calls == [
            {
                "prompt": "a cat",
                "n_prompt": "blurry",
                "height": 512,
                "width": 768,
                "steps": 20,
                "seed": 7,
                "num_frames": 121,
                "fps": 24,
                "guidance_scale": 5.0,
                "guidance_scale_2": 3.5,
                "use_image_aspect": False,
                "use_upscaler": True,
                "use_face_restore": True,
                "image_bytes": b"png-bytes",
            },
        ]

    def test_run_inference_accepts_no_image(
        self, fake_from_name: Callable[[object], FakeCls]
    ) -> None:
        fake = fake_from_name(b"mp4")
        ti2v = new_ti2v(
            prompts=make_video_prompts(),
            num_frames=81,
            fps=16,
            guidance_scale=5.0,
            guidance_scale_2=None,
            use_image_aspect=True,
            use_upscaler=False,
            use_face_restore=False,
        )

        ti2v.run_inference(Seed(1), None)

        assert fake.run_inference.calls[0]["image_bytes"] is None
        assert fake.run_inference.calls[0]["guidance_scale_2"] is None
