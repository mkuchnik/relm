"""A generic strategy and API for executing relm queries.

Copyright (C) 2023 Michael Kuchnik. All Right Reserved.
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple, Union

from relm.relm_search_query import SearchQuery


@dataclass
class QueryResult:
    """A wrapper around query results.

    tokens: The matching tokens for a result.
    score: The log probability of the result.
    """

    tokens: Tuple[int] = field(default_factory=lambda: [])
    score: Optional[float] = None


class SearchStrategy(ABC):
    """Search a test_relm abstractly."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Union[Iterable[Tuple[int]],
                                                  Iterable[QueryResult]]:
        """Search the model according to the pattern.
        :param query: A ReLM search query description.
        :returns: An iterator over matching results.
        """
        raise NotImplementedError("Search is not implemented.")
