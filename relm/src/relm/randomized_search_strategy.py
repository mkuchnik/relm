"""Define a randomize search strategy."""

import numpy as np

import relm.regex
import relm.relm_logging
import relm.util
from relm.abstract_search_strategy import SearchStrategy
from relm.relm_search_query import SearchQuery

logger = relm.relm_logging.get_logger()

# TODO(mkuchnik): Consider setting this depending on CPU/GPU availability
DEFAULT_TUNING_NUM_SAMPLES = 1024


class RandomizedSearchStrategy(SearchStrategy):
    """Search a test_relm randomly."""

    def __init__(self, test_relm: relm.model_wrapper.TestableModel):
        """Take a test_relm object."""
        assert isinstance(test_relm, relm.model_wrapper.TestableModel)
        self.test_relm = test_relm

    def search(self, query: SearchQuery):
        """Search the model according to the pattern."""
        logger.debug("Searching query: {}".format(query))
        if query.force_batch_tuning:
            logger.info("Starting batch autotuning")
            batch_size = _find_best_batch_size(self.test_relm, query)
        else:
            batch_size = None

        opt_flags = relm.regex.OptimizationFlags()
        opt_flags.top_k_sampling = query.top_k_sampling
        opt_flags.truncate_max_tokens = False
        it = relm.regex.sampled_regex_search_iterator(
            self.test_relm, query.query_str, batch_size=batch_size,
            return_probabilities=False,
            return_tokens=True,
            optimization_flags=opt_flags,
            proxy_model=None,
            test_relm_confirm_fn=None)
        # Get tokens
        search_iter = map(lambda x: tuple(x[-1].cpu().numpy()), it)
        logger.info("Starting sampling iterator.")
        ret = relm.util.sample_iterator(search_iter,
                                        query.num_samples,
                                        wrap_pbar=query.progress_bar)

        def log_entry(x):
            logger.info("Yielded sample: {}".format(x))
            return x
        ret = map(log_entry, ret)
        if query.num_samples:
            ret = list(ret)
            logger.debug("Returned {}".format(ret))
        else:
            # TODO(mkuchnik): Similarly wrap with pbar
            logger.info("Yielding iterator")
        return ret


def _find_best_batch_size(test_relm, query: SearchQuery, num_samples=None,
                          batch_sizes=None):
    """Perform a sweep over batch sizes to find the fastest one.

    Runs over num_samples is specified.
    """
    rates = []
    opt_flags = relm.regex.OptimizationFlags()
    opt_flags.replace_badmatch_with_none = True
    if num_samples is None:
        num_samples = DEFAULT_TUNING_NUM_SAMPLES
    if batch_sizes is None:
        batch_sizes = [2**i for i in range(0, 14)]
    batch_sizes = sorted(batch_sizes, reverse=True)
    rates = np.array([-100. for b in batch_sizes])
    # TODO(mkuchnik): Add a burn-in bench for first non-OOM run.
    for i, b in enumerate(batch_sizes):
        it = relm.regex.sampled_regex_search_iterator(
            test_relm, query.query_str, batch_size=b,
            return_probabilities=False,
            optimization_flags=opt_flags, proxy_model=None,
            test_relm_confirm_fn=None)
        try:
            rate = relm.util.benchmark_iterator(
                it,
                num_samples=num_samples if num_samples >= b else b,
                wrap_pbar=False)
        except RuntimeError as ex:
            logger.error(ex)
            if _is_cuda_memory_error(ex):
                continue
            else:
                raise ex
        logger.info("Rate at batch_size={}: {}".format(b, rate))
        rates[i] = rate
        if i and rates[i - 1] >= rates[i]:
            logger.info("Early tuning termination ({} >= {}).".format(
                rates[i - 1], rates[i]))
            break
    best_idx = np.argmax(rates)
    assert rates[best_idx] >= 0
    best_batch_size = batch_sizes[best_idx]
    logger.info("Found best batch size: {}".format(best_batch_size))
    return best_batch_size


def _is_cuda_memory_error(ex) -> bool:
    """Return true if exception is OOM CUDA error."""
    return (isinstance(ex, RuntimeError) and
            "CUDA out of memory." in str(ex))
