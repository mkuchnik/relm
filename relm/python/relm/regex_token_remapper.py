"""Define remappers for converting tokens between spaces.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod

import relm.relm_logging

logger = relm.relm_logging.get_logger()


class RegexTokenRemapper(ABC):
    """Define a class for mapping tokens to a different space."""

    @abstractmethod
    def encode(self, token):
        """Put token in new space."""
        raise NotImplementedError("Encode is not implemented.")

    @abstractmethod
    def decode(self, token):
        """Put token in original space."""
        raise NotImplementedError("Decode is not implemented.")


class OffsetTokenRemapper(ABC):
    """Map tokens by a constant offset."""

    def __init__(self, offset: int):
        """Initialize with offset."""
        self.offset = offset

    def encode(self, token):
        """Put tokens in new space."""
        if token >= self.offset:
            logger.warning("Token {} >= offset {}".format(token, self.offset))
        return token + self.offset

    def decode(self, token):
        """Put token in original space."""
        if token < self.offset:
            logger.warning("Token {} < offset {}".format(token, self.offset))
        return token - self.offset
