# grpo_project/callbacks/base.py
from transformers import TrainerCallback
import logging
import os

logger = logging.getLogger(__name__)

class BaseCallback(TrainerCallback):
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir
        if self.output_dir:
            # Ensure the output directory for the callback exists
            # Specific subdirectories can be created by subclasses if needed
            os.makedirs(self.output_dir, exist_ok=True)
        # self.logger = self._setup_logger() # Specific loggers can be setup in subclasses

    # def _setup_logger(self):
    #     # Placeholder for potential shared logger setup
    #     # For example, could create a logger specific to this callback instance
    #     # and save logs to a file within self.output_dir
    #     pass

    def on_train_begin(self, args, state, control, **kwargs):
        if self.output_dir:
            logger.info(f"{self.__class__.__name__}: Output directory set to {self.output_dir}")
        else:
            logger.info(f"{self.__class__.__name__}: No output directory provided.")
