"""A generic strategy for executing relm queries."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from relm.relm_search_query import SearchQuery


@dataclass
class QueryResult:
    """A wrapper around query results."""

    tokens: List = field(default_factory=lambda: [])
    score: Optional[float] = None


class SearchStrategy(ABC):
    """Search a test_relm abstractly."""

    @abstractmethod
    def search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        raise NotImplementedError("Search is not implemented.")
