import logging
from contextlib import contextmanager
from typing import Optional

# This global-like variable will hold the logger for the currently active build process.
_current_logger: Optional[logging.Logger] = None
_default_logger = logging.getLogger('model_builder_fallback')
_default_logger.addHandler(logging.StreamHandler())
_default_logger.setLevel(logging.WARNING)


def set_current_logger(logger: Optional[logging.Logger]):
    """
    Sets the logger for the current execution context.
    """
    global _current_logger
    _current_logger = logger

def get_current_logger() -> logging.Logger:
    """
    Retrieves the logger set for the current context.
    
    If no logger is set (e.g., when running a module standalone),
    it returns a default fallback logger.
    """
    global _current_logger
    if _current_logger is None:
        return _default_logger
    return _current_logger

@contextmanager
def user_logger_context(logger: logging.Logger):
    """
    A context manager to safely set and unset the global logger
    for the duration of a user's build pipeline.
    """
    set_current_logger(logger)
    try:
        yield
    finally:
        # Clear the logger once the context is exited
        set_current_logger(None)
