import logging
import sys
import os

def setup_global_logging(log_level: int, log_file_path: str, local_rank: int):
    """
    Sets up global logging configuration for the application.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file_path: Path to the log file.
        local_rank: Local process rank for distributed training, used in log format.
    """
    log_handlers = [logging.StreamHandler(sys.stdout)]

    if local_rank <= 0:  # Typically, file logging is done only by the main process
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir: # Check if log_dir is not an empty string (i.e. log file is in current dir)
             os.makedirs(log_dir, exist_ok=True)

        # Determine log mode: 'a' for append if file exists (resuming), 'w' for write otherwise
        log_mode = "a" if os.path.exists(log_file_path) else "w"
        file_handler = logging.FileHandler(log_file_path, mode=log_mode, encoding='utf-8')
        log_handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=f"[RANK {local_rank:02d}] %(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
        handlers=log_handlers,
        force=True,  # Override any existing configuration
    )

    # Configure transformers logger
    transformers_logger = logging.getLogger("transformers")
    # Set transformers logger level: WARNING for main process, ERROR for others to reduce verbosity
    transformers_logger.setLevel(logging.WARNING if local_rank <= 0 else logging.ERROR)

    logger = logging.getLogger(__name__)
    logger.info(f"Global logging setup complete. Log level: {logging.getLevelName(log_level)}. Log file: {log_file_path if local_rank <=0 else 'N/A (non-main process)'}")
