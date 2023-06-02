"""Relm logging utilities.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

import logging

_logger = None


def init_logger():
    """Initialize the relm logger."""
    # TODO(mkuchnik): Add proper logger instance
    global _logger
    _logger = logging.getLogger('relm')
    _logger.addHandler(logging.NullHandler())


def get_logger():
    """Get the relm logger."""
    if not _logger:
        init_logger()
    assert _logger
    return _logger
