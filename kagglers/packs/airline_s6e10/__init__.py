from pathlib import Path

from kagglers.packs.airline_s6e10.config import AirlineConfig
from kagglers.packs.airline_s6e10.pack import NAME, AirlinePack

PACK_NAME = NAME


def build_pack(config: AirlineConfig, workdir: Path) -> AirlinePack:
    return AirlinePack(config, workdir)


PACKS = {PACK_NAME: build_pack}
