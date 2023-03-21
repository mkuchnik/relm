"""Wrapper around native bindings for regex."""

from dataclasses import dataclass
from typing import Optional

import pywrapfst as fst
import rust_regex_compiler_bindings

import relm.automata
import relm.relm_logging
from relm.regex_token_remapper import RegexTokenRemapper

logger = relm.relm_logging.get_logger()


@dataclass
class FSTEdgeTransition:
    """Represent an FST transition from a state to another state."""

    state_0: int
    state_1: int
    transition_in: int
    transition_out: int
    weight: float


@dataclass
class FSTStateAcceptor:
    """Represent an FST state which may accept and emit a weight."""

    state_0: int
    weight: float


def _remap_edge_transition(
    edge_transition: FSTEdgeTransition,
        token_remapper: Optional[RegexTokenRemapper] = None):
    if token_remapper:
        # NOTE(mkuchnik): We can get high transition_in and transition_out
        # due to wildcards
        edge_transition.transition_in = \
            token_remapper.encode(edge_transition.transition_in)
        edge_transition.transition_out = \
            token_remapper.encode(edge_transition.transition_out)
    return edge_transition


def _unpack_fst_str_to_automata(
    fst_str: str, delimiter: str = "\t", verify: bool = True,
        token_remapper: Optional[RegexTokenRemapper] = None):
    """Return the openfst automata representing the string."""
    # TODO(mkuchnik): Compare against fst compile
    # https://www.openfst.org/twiki/bin/view/FST/FstQuickTour#CreatingShellFsts
    logger.debug("Unpacking Rust fst:\n'{}'".format(fst_str))
    lines = fst_str.split("\n")
    states = set()
    edge_transitions = []
    state_transitions = []
    symbols = set()
    for line in lines:
        vals = line.split(delimiter)
        try:
            if len(vals) == 5:
                # Connection information
                # Format:
                # state_0 state_1 transition_in transition_out weight
                state_0, state_1, transition_in, transition_out, weight = vals
                state_0 = int(state_0)
                state_1 = int(state_1)
                transition_in = int(transition_in)
                transition_out = int(transition_out)
                weight = int(weight)
                edge_transition = FSTEdgeTransition(
                    state_0, state_1, transition_in, transition_out, weight)
                edge_transition = _remap_edge_transition(edge_transition,
                                                         token_remapper)
                edge_transitions.append(edge_transition)
                # NOTE(mkuchnik): We have to use member variable for remap
                symbols.add(edge_transition.transition_in)
                symbols.add(edge_transition.transition_out)
                states.add(state_0)
                states.add(state_1)
            elif len(vals) == 2:
                # Node acceptor information
                # state_0 weight
                state_0, weight = vals
                state_0 = int(state_0)
                weight = int(weight)
                state_transition = FSTStateAcceptor(state_0, weight)
                state_transitions.append(state_transition)
                states.add(state_0)
            elif len(vals) == 0:
                # Black line
                continue
            elif len(vals) == 1 and not vals[0]:
                # Empty string
                continue
            else:
                raise ValueError("Unknown values: {}".format(vals))
        except Exception as ex:
            logger.error("Failed unpacking line: '{}'".format(line))
            raise ex

    f = fst.VectorFst()
    f.reserve_states(len(states))

    for _ in states:
        emitted_state = f.add_state()
        assert emitted_state in states

    f.set_start(0)   # Assume 0 is the start state

    # Transition states
    for edge_transition in edge_transitions:
        f.add_arc(
            edge_transition.state_0,
            fst.Arc(edge_transition.transition_in,
                    edge_transition.transition_out,
                    edge_transition.weight,
                    edge_transition.state_1))

    # Let's set all states to have non-accepting final weight
    for state_acceptor in state_transitions:
        f.set_final(state_acceptor.state_0, state_acceptor.weight)

    if verify and not f.verify():
        raise ValueError("Automata failed verification")

    if token_remapper:
        # NOTE(mkuchnik): We do not have to encode s as it's already encoded
        # But, we have to decode before casting to char
        symbol_table = {s: chr(token_remapper.decode(s)) for s in symbols}
    else:
        symbol_table = {s: chr(s) for s in symbols}

    f = relm.automata.attach_symbol_table(f, symbol_table)

    return f


def regex_to_automata(regex: str,
                      token_remapper: Optional[RegexTokenRemapper] = None):
    """Return the openfst automata representing the regex."""
    logger.debug("Starting Rust FST parsing")
    fst_str = rust_regex_compiler_bindings.regex_to_fst(regex)
    logger.debug("Finished Rust FST parsing")
    fst = _unpack_fst_str_to_automata(fst_str,
                                      token_remapper=token_remapper)
    logger.debug("Finished unpacking Rust FST payload")
    return fst
