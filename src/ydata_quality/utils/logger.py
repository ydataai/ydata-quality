import logging
from typing import TextIO
import sys
import os

# Default vars for the logger
NAME = os.getenv('DQ_LOGGER_NAME', 'DQ_Logger')
STREAM = sys.stdout
LOG_LEVEL = os.getenv('DQ_LOG_LEVEL', logging.WARNING)

def create_logger(name, stream: TextIO = sys.stdout, level=logging.INFO):
  handler = logging.StreamHandler(stream)
  handler.setFormatter(
      logging.Formatter(
          "%(levelname)s | %(message)s"
      )
  )

  logger = logging.getLogger(name)
  logger.setLevel(level)
  if len(logger.handlers)==0:
    logger.addHandler(handler)
  logger.propagate = False

  return logger
