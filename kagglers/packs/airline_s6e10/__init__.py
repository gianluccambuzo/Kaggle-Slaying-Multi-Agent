from pathlib import Path

from kagglers.core.sandbox import SandboxLimits
from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.pack import DEFAULT_LIMITS, AirlinePack


def build_pack(
    config: AirlineConfig, workdir: Path, limits: SandboxLimits | None = None
) -> AirlinePack:
    """The single constructor of the pack; `kagglers.packs.get_pack` goes through it."""
    return AirlinePack(config, workdir, limits or DEFAULT_LIMITS)
