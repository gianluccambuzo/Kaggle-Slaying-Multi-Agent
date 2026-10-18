"""Pack registry: look up a competition pack by name."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from kagglers.core.types import Pack


class UnknownPackError(KeyError):
    def __init__(self, name: str, available: list[str]):
        self.name = name
        self.available = available
        super().__init__(name)

    def __str__(self) -> str:
        return f"unknown pack {self.name!r}; available: {', '.join(self.available)}"


def _airline_s6e10(workdir: Path = Path("."), limits=None, **config_kwargs) -> Pack:
    from kagglers.packs import airline_s6e10
    from kagglers.packs.airline_s6e10.config import AirlineConfig

    workdir = Path(workdir)
    # The default data_dir is relative ("data/<competition>"): anchor it at workdir.
    config_kwargs.setdefault("data_dir", workdir / AirlineConfig.data_dir)
    return airline_s6e10.build_pack(AirlineConfig(**config_kwargs), workdir, limits)


_REGISTRY: dict[str, Callable[..., Pack]] = {"airline_s6e10": _airline_s6e10}


def available_packs() -> list[str]:
    return sorted(_REGISTRY)


def get_pack(name: str, **kwargs) -> Pack:
    """Build the pack `name`. kwargs: `workdir`, `limits` (SandboxLimits) and the
    pack's config fields (for airline_s6e10: data_dir, positive_label, ...)."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownPackError(name, available_packs()) from None
    return factory(**kwargs)
