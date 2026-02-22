"""Logging configuration for the app."""

from __future__ import annotations

import logging
import warnings

from .config import AppConfig


def setup_logging(config: AppConfig) -> logging.Logger:
    logger = logging.getLogger("kokoro_app")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.log_level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setLevel(config.file_log_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | "
            "%(funcName)s | %(message)s"
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.DEBUG)
    warnings_logger.propagate = False
    for handler in list(warnings_logger.handlers):
        warnings_logger.removeHandler(handler)
    warnings_logger.addHandler(file_handler)

    hf_logger = logging.getLogger("huggingface_hub")
    hf_logger.setLevel(logging.DEBUG)
    hf_logger.propagate = False
    for handler in list(hf_logger.handlers):
        hf_logger.removeHandler(handler)
    hf_logger.addHandler(file_handler)
    return logger
