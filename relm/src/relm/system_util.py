"""Functions used to inspect and modify system resources."""

import psutil


def available_memory():
    """Return the amount of available system memory."""
    return psutil.virtual_memory().available
