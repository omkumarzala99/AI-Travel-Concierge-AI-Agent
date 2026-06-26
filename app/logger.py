import logging
import time
from functools import wraps
from typing import Callable, Any

# Configure logging formatting and output level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Root app logger
logger = logging.getLogger("AI-Travel-Concierge")

def get_logger(name: str) -> logging.Logger:
    """Get a scoped logger under the main app namespace."""
    return logging.getLogger(f"AI-Travel-Concierge.{name}")

def log_execution_time(logger_name: str = "performance"):
    """
    Decorator to measure execution time of a function.
    Logs start, completion, duration in seconds, and handles error boundaries.
    """
    perf_logger = get_logger(logger_name)
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            perf_logger.info(f"Starting execution of '{func.__name__}'")
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                perf_logger.info(f"Finished execution of '{func.__name__}' in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                perf_logger.error(
                    f"Exception in '{func.__name__}' after {duration:.4f}s: {str(e)}", 
                    exc_info=True
                )
                raise
        return wrapper
    return decorator
