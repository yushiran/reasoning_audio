# config.py
import os
from pathlib import Path


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", None)

global_config = Config()