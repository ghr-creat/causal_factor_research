"""Centralized logging configuration for the causal factor research project."""
import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_dir: Path = None,
    log_file: str = "causal_pipeline.log",
    level: str = "INFO",
    retention: str = "7 days",
    rotation: str = "10 MB",
):
    """Configure loguru with console and file sinks.

    Parameters
    ----------
    log_dir : Path
        Directory to store log files. Defaults to results/causal_final/logs.
    log_file : str
        Log filename.
    level : str
        Minimum log level.
    retention : str
        How long to keep log files.
    rotation : str
        Log rotation policy.
    """
    if log_dir is None:
        # Avoid circular import by not importing config at module load time
        log_dir = Path(__file__).resolve().parent / "results" / "causal_final" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_file

    # Remove default sink to avoid duplicates if called multiple times
    logger.remove()
    # Console sink
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        enqueue=False,
    )
    # File sink with rotation
    logger.add(
        log_path,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
    )
    logger.info(f"Logging configured. Log file: {log_path}")
    return logger


if __name__ == "__main__":
    setup_logging()
    logger.info("Logging test message")
