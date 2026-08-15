"""训练时 CUDA 约束和模块级日志工具。"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from base.logger import get_logger


@dataclass(frozen=True)
class CudaInfo:
    """一次训练使用的 CUDA 环境信息。"""

    device_name: str
    torch_version: str
    cuda_version: str | None
    total_memory_bytes: int


def require_cuda() -> CudaInfo:
    """强制要求 CUDA，禁止训练静默降级到 CPU。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for query-router training, but torch.cuda.is_available() is False.")
    properties = torch.cuda.get_device_properties(0)
    return CudaInfo(
        device_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        total_memory_bytes=int(properties.total_memory),
    )


def configure_training_logger(log_directory: Path) -> logging.Logger:
    """为训练过程额外写入模块独立日志文件。"""
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = get_logger("model_trian_classify")
    log_path = log_directory / "train.log"
    if not any(getattr(handler, "baseFilename", None) == str(log_path) for handler in logger.handlers):
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.name = "query_router_file_handler"
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def cuda_log_payload(cuda_info: CudaInfo) -> dict[str, int | str | None]:
    """返回可写入 JSON 指标文件的 CUDA 信息。"""
    return asdict(cuda_info)
