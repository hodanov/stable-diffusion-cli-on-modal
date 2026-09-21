from __future__ import annotations

from typing import TYPE_CHECKING, Any

import modal
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def raw_entrypoint() -> Callable[[modal.app.LocalEntrypoint], Callable[..., Any]]:
    """Fixture returning the unwrapper below."""
    return unwrap_local_entrypoint


def unwrap_local_entrypoint(
    entrypoint: modal.app.LocalEntrypoint,
) -> Callable[..., Any]:
    """
    Return the undecorated function behind a Modal local entrypoint.

    `@app.local_entrypoint()` replaces the function with a LocalEntrypoint that
    needs a running app, so tests call the raw function instead. Fail loudly if
    a Modal upgrade moves it, rather than silently testing something else.
    """
    raw_f = getattr(entrypoint.info, "raw_f", None)
    if raw_f is None:  # pragma: no cover - only hit on a Modal API change
        msg = f"Could not unwrap the local entrypoint (modal {modal.__version__}). Update raw_entrypoint()."
        raise AssertionError(msg)
    return raw_f
