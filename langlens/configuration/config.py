import logging.config
from logging import Logger
from pathlib import Path

import yaml

# App Config
with open(Path(__file__).parent / "app.yaml", "r") as file:
    settings = yaml.safe_load(file)

# Logging Config
logging.config.fileConfig(Path(__file__).parent / "logging.ini", disable_existing_loggers=False)


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(f"langlens.{name}")
    return logger
