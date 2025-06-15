import os
import logging
import numpy as np
import torch
import gc
import sys

# --- BEGIN: PyTorch Safe Unpickling Configuration ---
# This should run once at the beginning of the application.
# logger_temp is used for logging during this initial setup phase.
logger_temp = logging.getLogger(__name__ + "_startup")
try:
    from numpy.core.multiarray import _reconstruct
    from numpy import dtype as numpy_dtype
    from numpy.dtypes import UInt32DType

    safe_globals_list = []
    if callable(_reconstruct): safe_globals_list.append(_reconstruct)
    else: logger_temp.warning("_reconstruct is not callable.")
    if isinstance(numpy_dtype, type): safe_globals_list.append(numpy_dtype)
    else: logger_temp.warning("numpy.dtype is not a type.")
    if isinstance(np.ndarray, type): safe_globals_list.append(np.ndarray)
    else: logger_temp.warning("np.ndarray is not a type.")
    if isinstance(UInt32DType, type): safe_globals_list.append(UInt32DType)
    else: logger_temp.warning("UInt32DType is not a type.")

    numpy_scalar_types_to_add = [
        np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
        np.int_, np.uint, np.longlong, np.ulonglong,
        np.half, np.float16, np.single, np.double, np.longdouble,
        np.csingle, np.cdouble, np.clongdouble,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ]
    added_scalar_types_count = 0
    for nt_class in numpy_scalar_types_to_add:
        if isinstance(nt_class, type):
            safe_globals_list.append(nt_class)
            added_scalar_types_count += 1
        else:
            logger_temp.warning(f"NumPy scalar '{str(nt_class)}' (type: {type(nt_class)}) is not directly a type class, not adding.")

    if safe_globals_list:
        torch.serialization.add_safe_globals(safe_globals_list)
        logger_temp.info(f"Successfully updated torch safe global variables list with {len(safe_globals_list)} items, including {added_scalar_types_count} scalar types.")
    else:
        logger_temp.warning("safe_globals_list is empty before calling torch.serialization.add_safe_globals.")

except ImportError as e:
    logger_temp.warning(f"Failed to import NumPy modules for torch safe globals: {e}.")
except AttributeError as e:
    logger_temp.warning(f"Attribute error accessing NumPy properties for torch safe globals: {e}.")
except Exception as e_globals:
    logger_temp.error(f"An unexpected error occurred while setting up torch safe globals: {e_globals}", exc_info=True)
# --- END: PyTorch Safe Unpickling Configuration ---

# Main entry point for the training script
if __name__ == "__main__":
    # The TrainingOrchestrator and its CLI function handle argument parsing and all training logic.
    # We import it here to make train.py the main executable script.
    try:
        from grpo_project.core.trainer import main_orchestrator_cli
        main_orchestrator_cli()
    except ImportError as e_orch:
        # Use a basic logger if the main logging from orchestrator isn't set up
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"Failed to import TrainingOrchestrator or its CLI: {e_orch}", exc_info=True)
        logging.error("Please ensure 'grpo_project.core.trainer' is correctly structured and all dependencies are met.")
        sys.exit(1)
    except Exception as e_cli:
        # Catch any other exception during the CLI execution that wasn't handled by the orchestrator
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"An error occurred during the training process via main_orchestrator_cli: {e_cli}", exc_info=True)
        sys.exit(1)
    finally:
        # Final cleanup, especially for CUDA memory, if torch was initialized.
        if 'torch' in sys.modules and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger_temp.info("Final CUDA cache clear attempted.") # Use logger_temp or a default logger
            except Exception as e_cuda_final:
                logger_temp.warning(f"Error during final CUDA cache clear: {e_cuda_final}")
        if 'gc' in sys.modules:
            gc.collect()
            logger_temp.info("Final garbage collection attempted.")
        
        # Ensure all logging handlers are flushed, especially if writing to files.
        # This might be more robust if the orchestrator's logger is accessible here,
        # or if a global logging shutdown function is implemented.
        if logging.getLogger().hasHandlers():
            for handler in logging.getLogger().handlers:
                try:
                    handler.flush()
                except Exception:
                    pass # Ignore errors on flushing, e.g. if handler was closed
        
        logger_temp.info("train.py script execution finished.")
