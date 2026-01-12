import enum
from dataclasses import dataclass


class ProcessingMode(str, enum.Enum):
    SEQUENTIAL = 'sequential'
    BATCH = 'batch'
    MULTIPROCESSING = 'multiprocessing'


@dataclass
class ProcessingCapabilties:
    """Capabilities of an explanation method."""
    supports_batch: bool = False
    supports_mp: bool = False
