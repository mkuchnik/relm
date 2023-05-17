"""Define how a query runs."""
import enum
from dataclasses import dataclass
from typing import Callable, List, Optional

from relm.regex_parser_strategy import AutomataPreprocessor


@dataclass
class PrepareOptions:
    """Flags to enable model preparation features."""

    manage_device_placement: bool = True


class SearchBackendType(enum.Enum):
    """Determine the backend for searching."""

    RANDOMIZED = enum.auto()
    AUTOMATA = enum.auto()


class RegexBackendType(enum.Enum):
    """Determine the backend for regex."""

    PYTHON = enum.auto()
    RUST = enum.auto()


@dataclass
class SearchQuery:
    """Flags to enable search features along with query itself.

    This is the low-level API.

    query_str: is the query itself
    num_samples: The number of sampling to return (or infinite).
    progress_bar: is a visual progress bar
    force_batch_tuning: makes the batch size automatically a good default
    backend: one of the SearchBackendTypes
    top_k_sampling: Whether to use top-k sampling and what value.
    temperature: Whether to use temperature scaling during sampling and what
    value.
    sequence_length: The maximum length of token sequences to consider. If not
    set, defaults to model maximum.
    """

    query_str: str
    accept_str: Optional[str] = None
    num_samples: int = 1
    progress_bar: bool = True
    force_batch_tuning: bool = False
    backend: SearchBackendType = SearchBackendType.RANDOMIZED
    top_k_sampling: Optional[int] = None
    temperature: Optional[float] = None
    sequence_length: Optional[int] = None
    experimental_cache_optimization: Optional[bool] = None
    experimental_IDDFS_optimization: Optional[bool] = None
    experimental_log_k: Optional[bool] = None
    experimental_query_automata: Optional[bool] = None
    experimental_accept_automata: Optional[bool] = None
    experimental_fast_start: Optional[bool] = None
    experimental_very_fast_start: Optional[bool] = None
    experimental_dijkstra: Optional[bool] = None
    experimental_dijkstra_aggregation_function: Optional[bool] = None
    experimental_dijkstra_beam_size: Optional[bool] = None
    experimental_greedy_search: Optional[bool] = None
    experimental_safe: Optional[bool] = None
    experimental_advanced_parsing: Optional[bool] = None
    experimental_advanced_parsing_simplify: Optional[bool] = True
    experimental_advanced_parsing_static_minimize: Optional[bool] = False
    experimental_advanced_parsing_static_minimize_prefix_only: Optional[bool] \
        = False
    experimental_advanced_parsing_dynamic_minimize: Optional[bool] = False
    experimental_add_eos_token: Optional[bool] = None
    experimental_inner_query_callback: Optional[Callable[[], None]] = None
    experimental_penalized_accepted_probability: Optional[bool] = None
    experimental_avoid_not_accepted_probability: Optional[bool] = None
    experimental_regex_backend: RegexBackendType = RegexBackendType.PYTHON
    experimental_automata_preprocessors: Optional[List[AutomataPreprocessor]] \
        = None
    experimental_random_sampling: Optional[bool] = None
    experimental_truncate_automata: Optional[bool] = None
    experimental_advanced_results: Optional[bool] = None
    experimental_filter_str: Optional[str] = None
    experimental_random_sampling_normalization: Optional[bool] = True

    @staticmethod
    def _expected_slots():
        # TODO(mkuchnik): consider caching
        expected_slots = \
            ("query_str",
             "accept_str",
             "num_samples",
             "progress_bar",
             "force_batch_tuning",
             "backend",
             "top_k_sampling",
             "temperature",
             "sequence_length",
             "experimental_cache_optimization",
             "experimental_IDDFS_optimization",
             "experimental_log_k",
             "experimental_query_automata",
             "experimental_accept_automata",
             "experimental_fast_start",
             "experimental_very_fast_start",
             "experimental_dijkstra",
             "experimental_dijkstra_aggregation_function",
             "experimental_dijkstra_beam_size",
             "experimental_greedy_search",
             "experimental_safe",
             "experimental_advanced_parsing",
             "experimental_advanced_parsing_simplify",
             "experimental_advanced_parsing_static_minimize",
             "experimental_advanced_parsing_static_minimize_prefix_only",
             "experimental_advanced_parsing_dynamic_minimize",
             "experimental_add_eos_token",
             "experimental_inner_query_callback",
             "experimental_penalized_accepted_probability",
             "experimental_avoid_not_accepted_probability",
             "experimental_regex_backend",
             "experimental_automata_preprocessors",
             "experimental_random_sampling",
             "experimental_truncate_automata",
             "experimental_advanced_results",
             "experimental_filter_str",
             "experimental_random_sampling_normalization",
             )
        return expected_slots

    def to_search_query(self) -> 'SearchQuery':
        """Return a search query."""
        return self

    def __setattr__(self, k, v):
        """Limit the setattr function to known attributes."""
        if k not in SearchQuery._expected_slots():
            raise AttributeError(
                "{} dataclass has no field {}. Must be one of: {}".format(
                    type(self), k, SearchQuery._expected_slots()))
        super().__setattr__(k, v)


@dataclass
class QueryString:
    """A compact and type-safe representation of a query.

    query_str: The query to match on (regex)
    prefix_str: The prefix of the query to match on (regex)
    add_eos: Add EOS token to query_str if True
    """

    query_str: str
    prefix_str: Optional[str] = None
    add_eos: Optional[bool] = None


class QuerySearchStrategy(enum.Enum):
    """Determine the type of search query to use during decoding.

    shortest_path: The most likely answers according to model.
    random_sampling: Sample answers with model-dependent probability.
    """

    SHORTEST_PATH = enum.auto()
    RANDOM_SAMPLING = enum.auto()


class QueryTokenizationStrategy(enum.Enum):
    """Determine the type of tokenization to enforce over query strings.

    all_tokens: Return all tokens matching the query string.
    canonical_tokens: Return only canonical tokens matching the query string.
    prefix_canonical_tokens: Return only canonical tokens in the prefix and
    allow anything afterwords.
    """

    ALL_TOKENS = enum.auto()
    CANONICAL_TOKENS = enum.auto()
    PREFIX_CANONICAL_TOKENS = enum.auto()


@dataclass
class QueryPreprocessors:
    """Query modifiers to capture edits and filters.

    automata_preprocessors: Automata preprocessors (e.g., for edits)
    filter_str: The query to not match on (regex)
    """

    automata_preprocessors: Optional[List[AutomataPreprocessor]] = None
    filter_str: Optional[str] = None


@dataclass
class SimpleSearchQuery:
    """A front-end to Search Query.

    Most users should go through this.
    query_string: A QueryString object describing what strings to look for.
    search_strategy: A QuerySearchStrategy object describing the type of
    sampling to use (e.g., best-first or random).
    tokenization_strategy: A QueryTokenizationStrategy object describing the
    types of tokens to consider (e.g., all or canonical).
    top_k_sampling: Whether to use top-k sampling and what value.
    temperature: Whether to use temperature scaling during sampling and what
    value.
    sequence_length: The maximum length of token sequences to consider. If not
    set, defaults to model maximum.
    num_samples: The number of sampling to return (or infinite).
    progress_bar: Displays a tqdm progress bar if True.
    preprocessors: A QueryPreprocessors object describing additional modifiers
    to apply to query_string, such as edits or filters.
    """

    query_string: QueryString
    search_strategy: QuerySearchStrategy
    tokenization_strategy: QueryTokenizationStrategy
    top_k_sampling: Optional[int] = None
    temperature: Optional[float] = None
    sequence_length: Optional[int] = None
    num_samples: Optional[int] = None
    progress_bar: bool = False
    preprocessors: Optional[QueryPreprocessors] = None

    def to_search_query(self) -> 'SearchQuery':
        """Return a search query.

        Translates this API to low-level API.
        """
        query = SearchQuery(
            query_str=self.query_string.query_str,
            accept_str=self.query_string.prefix_str,
            num_samples=self.num_samples,
            progress_bar=True,
            force_batch_tuning=False,
            backend=SearchBackendType.AUTOMATA,
            top_k_sampling=self.top_k_sampling,
            temperature=self.temperature,
            sequence_length=self.sequence_length,
            experimental_cache_optimization=None,
            experimental_IDDFS_optimization=None,
            experimental_log_k=None,
            experimental_query_automata=None,
            experimental_accept_automata=None,
            experimental_fast_start=True,
            experimental_very_fast_start=True,
            experimental_dijkstra=(
                self.search_strategy ==
                QuerySearchStrategy.SHORTEST_PATH),
            experimental_dijkstra_aggregation_function=None,
            experimental_dijkstra_beam_size=None,
            experimental_greedy_search=None,
            experimental_safe=None,
            experimental_advanced_parsing=True,
            experimental_advanced_parsing_simplify=True,
            experimental_advanced_parsing_static_minimize=(
                self.tokenization_strategy ==
                QueryTokenizationStrategy.CANONICAL_TOKENS or
                self.tokenization_strategy ==
                QueryTokenizationStrategy.PREFIX_CANONICAL_TOKENS
            ),
            experimental_advanced_parsing_static_minimize_prefix_only=(
                self.tokenization_strategy ==
                QueryTokenizationStrategy.PREFIX_CANONICAL_TOKENS),
            experimental_advanced_parsing_dynamic_minimize=False,
            experimental_add_eos_token=(
                self.query_string.add_eos or
                self.search_strategy ==
                QuerySearchStrategy.RANDOM_SAMPLING),
            experimental_inner_query_callback=None,
            experimental_penalized_accepted_probability=(
                self.search_strategy ==
                QuerySearchStrategy.SHORTEST_PATH),
            experimental_avoid_not_accepted_probability=(
                self.tokenization_strategy ==
                QueryTokenizationStrategy.PREFIX_CANONICAL_TOKENS),
            experimental_regex_backend=RegexBackendType.RUST,
            experimental_automata_preprocessors=(
                None if not self.preprocessors
                else self.preprocessors.automata_preprocessors),
            experimental_random_sampling=(
                self.search_strategy ==
                QuerySearchStrategy.RANDOM_SAMPLING),
            experimental_truncate_automata=(
                self.search_strategy ==
                QuerySearchStrategy.RANDOM_SAMPLING),
            experimental_advanced_results=None,
            experimental_filter_str=(
                None if not self.preprocessors
                else self.preprocessors.filter_str),
            experimental_random_sampling_normalization=True,
        )
        return query
