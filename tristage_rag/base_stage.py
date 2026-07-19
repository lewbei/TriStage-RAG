"""Base class for pipeline stages with shared GPU management."""
import logging
from .utils import get_device, clear_gpu_cache


class BaseStage:
    """Base class providing common device management and GPU cleanup."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device: str = "cpu"
        self.use_amp: bool = False

    def _init_device(self, device: str = "auto", use_gpu: bool = True) -> str:
        """Initialize and return the compute device."""
        self.device = get_device(device, use_gpu)
        return self.device

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory if using CUDA."""
        clear_gpu_cache(self.device)

    def __del__(self):
        try:
            self.clear_gpu_memory()
        except Exception:
            pass
