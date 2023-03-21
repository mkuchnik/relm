"""Facade design pattern masking API behind simple usage.

API is simple:
1. facade.search(model, pattern)
2. f = facade.prepare(model); f.search(pattern)
"""
from typing import Union

import torch

import relm.model_wrapper
import relm.relm_logging
# NOTE(mkuchnik): We patch namespace. Refactor this out later.
import relm.relm_search_query
from relm.directed_automata_search_strategy import \
    DirectedAutomataSearchStrategy
from relm.randomized_search_strategy import RandomizedSearchStrategy
from relm.relm_search_query import (PrepareOptions, SearchBackendType,
                                    SearchQuery, SimpleSearchQuery)
from relm.simple_automata_search_strategy import SimpleAutomataSearchStrategy

RegexBackendType = relm.relm_search_query.RegexBackendType

logger = relm.relm_logging.get_logger()

query_type = Union[SearchQuery, SimpleSearchQuery]


def _is_directed_query(query: query_type) -> bool:
    """Return true if query should be directed."""
    query = query.to_search_query()
    return (query.experimental_dijkstra
            or query.experimental_greedy_search
            or query.experimental_random_sampling)


class PreparedReLMModel:
    """Exposes a search method for queries on a model."""

    def __init__(self, test_relm: relm.model_wrapper.TestableModel):
        """Take a test_relm object."""
        assert isinstance(test_relm, relm.model_wrapper.TestableModel)
        self.test_relm = test_relm

    def _dispatch_directed_query_strategy(self, query: query_type):
        """Return the strategy selected by query."""
        query = query.to_search_query()
        if _is_directed_query(query):
            strategy = DirectedAutomataSearchStrategy(self.test_relm)
        else:
            strategy = SimpleAutomataSearchStrategy(self.test_relm)
        return strategy

    def search(self, query: query_type):
        """Search the model according to the pattern."""
        query = query.to_search_query()
        if query.backend == SearchBackendType.RANDOMIZED:
            strategy = RandomizedSearchStrategy(self.test_relm)
        elif query.backend == SearchBackendType.AUTOMATA:
            strategy = self._dispatch_directed_query_strategy(query)
        else:
            raise NotImplementedError("Backend {} is not supported.".format(
                query.backend))
        return strategy.search(query)

    def plan(self, query: query_type):
        """Return the query plan."""
        query = query.to_search_query()
        if query.backend == SearchBackendType.AUTOMATA:
            strategy = self._dispatch_directed_query_strategy(query)
        else:
            raise NotImplementedError("Backend {} is not supported.".format(
                query.backend))
        return strategy.plan(query)

    def plan_accept(self, query: query_type):
        """Return the accept plan."""
        query = query.to_search_query()
        if query.backend == SearchBackendType.AUTOMATA:
            strategy = self._dispatch_directed_query_strategy(query)
        else:
            raise NotImplementedError("Backend {} is not supported.".format(
                query.backend))
        return strategy.plan_accept(query)


def search(model, tokenizer, query: query_type,
           prepare_options: PrepareOptions = None):
    """Search a model for the query pattern."""
    query = query.to_search_query()
    logger.debug("Preparing model: {} with tokenizer: {}".format(
        model, tokenizer))
    f = prepare(model, tokenizer, prepare_options)
    search_results = f.search(query)
    return search_results


def prepare(model, tokenizer, prepare_options: PrepareOptions = None):
    """One-time setup for search."""
    if prepare_options and not isinstance(prepare_options, PrepareOptions):
        raise ValueError(
            "Expected prepare_options to be of type PrepareOptions. "
            "Got {}".format(type(prepare_options))
        )
    if prepare_options is None:
        prepare_options = PrepareOptions()
    if prepare_options.manage_device_placement:
        is_model_using_cuda = next(model.parameters()).is_cuda
        if torch.cuda.is_available() and not is_model_using_cuda:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info("Moving model to device: {}".format(device))
            device = torch.device(device)
            model.to(device, non_blocking=True)
        model = model.eval()
    test_relm = relm.model_wrapper.TestableModel(model, tokenizer)
    prepared_model = PreparedReLMModel(test_relm)
    logger.info("Prepared model: {} with options: {}".format(prepared_model,
                                                             prepare_options))
    return prepared_model
