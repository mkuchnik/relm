"""Functions used to inspect and modify system resources.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import psutil


def available_memory():
    """Return the amount of available system memory."""
    return psutil.virtual_memory().available
