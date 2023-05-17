"""Complicated (compiler-like) graph optimizations for regex."""
import itertools
import pprint
from typing import Optional

import pywrapfst as fst

import relm.relm_logging
from relm.regex_token_remapper import RegexTokenRemapper

logger = relm.relm_logging.get_logger()


def minimize_canonical_automata(
        automata, test_relm, max_depth, full_vocab, trace=True,
        throw_on_collision=False,
        token_remapper: Optional[RegexTokenRemapper] = None,
        fast: bool = False):
    """Convert an automata into an equivalent canonical representation.

    The tokens used will be the same as given by Huggingface.
    We assume the automata does not have paths that yield equivalent words.
    """
    if fast:
        raise NotImplementedError("Fast not implemented.")
    else:
        new_automata = _minimize_canonical_automata_iterative_DFS(
            automata,
            test_relm,
            max_depth,
            full_vocab,
            trace=trace,
            throw_on_collision=throw_on_collision,
            token_remapper=token_remapper
        )
    return new_automata


def _minimize_canonical_automata_iterative_DFS(
        automata, test_relm, max_depth, full_vocab, trace=True,
        throw_on_collision=False,
        token_remapper: Optional[RegexTokenRemapper] = None):
    """Convert an automata into an equivalent canonical representation.

    The tokens used will be the same as given by Huggingface.
    We assume the automata does not have paths that yield equivalent words.
    """
    if relm.automata.is_cyclic_automata(automata):
        automata_summary = relm.automata.summarize_automata(automata)
        logger.warning("Automata is cyclic: {}".format(automata_summary))
        raise ValueError("Automata is cyclic: {}".format(automata_summary))

    dfs_it = relm.automata.DFS_from_automata(automata,
                                             return_edges_visited=True)
    all_accepted_tokens = []
    if trace:
        seen_words = dict()
    else:
        seen_words = set()
    # TODO(mkuchnik): This will block on infinite cardinality automata
    # TODO(mkuchnik): Use iterator
    for _, path in dfs_it:
        if token_remapper:
            remapped_path = tuple(map(token_remapper.decode, path))
        else:
            remapped_path = path
        logger.debug("Visiting DFS path: {} ({})".format(path, remapped_path))
        words = test_relm.tokens_to_words(remapped_path)
        if words in seen_words:
            if trace:
                original_path = seen_words[words]
                error_msg = (
                    "Found duplicate words: {} (path: {}, original: {})"
                    .format(
                        words, remapped_path, original_path)
                )
            else:
                error_msg = "Found duplicate words: {} (path: {})".format(
                        words, remapped_path
                )
            if throw_on_collision:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
        # TODO(mkuchnik): This is inefficient because it's not batched and it
        # creates PyTorch tensors
        tokens = test_relm.words_to_tokens(words).cpu().numpy()[0].tolist()
        if token_remapper:
            tokens = map(token_remapper.encode, tokens)
        tokens = tuple(tokens)
        if len(tokens) > max_depth:
            # TODO(mkuchnik): This is inefficient
            continue
        all_accepted_tokens.append(tokens)
        if trace:
            seen_words[words] = remapped_path
        else:
            seen_words.add(words)

    logger.info("Yielded {} token pairs in minimize automata.".format(
        len(all_accepted_tokens)))
    logger.info("Minimize automata samples: {}".format(
        pprint.pformat(list(itertools.islice(all_accepted_tokens, 100)))))

    if len(all_accepted_tokens):
        all_accepted_tokens = map(relm.automata.automata_from_token_list,
                                  all_accepted_tokens)
        new_automata = relm.automata.union_automatas(all_accepted_tokens)
    else:
        new_automata = relm.automata.null_automata()

    if token_remapper:
        inverted_full_vocab = {token_remapper.encode(v): k
                               for k, v in full_vocab.items()}
        relm.automata.attach_symbol_table(new_automata, inverted_full_vocab)
    else:
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        relm.automata.attach_symbol_table(new_automata, inverted_full_vocab)

    return new_automata


def simplify_automata_symbols(
        automata, full_vocab,
        token_remapper: Optional[RegexTokenRemapper] = None):
    """Add new edges where an equivalent one can be made with vocab."""
    return _simplify_automata_symbols_DFS(
        automata=automata,
        full_vocab=full_vocab,
        token_remapper=token_remapper)


def _simplify_automata_symbols_DFS(
        automata, full_vocab,
        token_remapper: Optional[RegexTokenRemapper] = None):
    """Add new edges where an equivalent one can be made with vocab.

    We iterate over k vocab items. For each one, we iterate over all
    edges. For each edge, we traverse the graph while matching the
    word. This gives us a complexity of O(k * V * |w|), where |w| is
    the max word size. Simply, we loop over all words, and from each word, we
    initiate a DFS path tracing search from all verticies.

    If there is a path on the word, then we can add an equivalent
    path (in terms of languages accepted). The path is equivalent to checking
    for the existence of a path in the extended transition function.
    """
    # NOTE(mkuchnik): We assume input symbols exists
    input_symbols = automata.input_symbols()
    if input_symbols is None:
        raise ValueError("Input symbols for automata is not set.")
    else:
        input_symbols = dict(input_symbols)

    # Fix meta-characters on large tokens
    inverted_full_vocab = {v: k for k, v in full_vocab.items()}
    assert len(inverted_full_vocab) == len(full_vocab), "Not 1-to-1 vocab"
    if token_remapper:
        input_symbols = {k: inverted_full_vocab[token_remapper.decode(k)]
                         for k in input_symbols.keys()}
    else:
        input_symbols = {k: inverted_full_vocab[k]
                         for k in input_symbols.keys()}

    inverted_input_symbols = {v: k for k, v in input_symbols.items()}
    # NOTE(mkuchnik): We cache the mapping in Python because arcs
    # calls are expensive.
    _arcs_cache = {s: [a for a in automata.arcs(s)] for s in automata.states()}

    # Primary key: ilabel
    # Secondary key: from_state
    _first_symbol_arcs_cache = dict()
    for s, arcs in _arcs_cache.items():
        for a in arcs:
            if a.ilabel not in _first_symbol_arcs_cache:
                _first_symbol_arcs_cache[a.ilabel] = dict()
            d = _first_symbol_arcs_cache[a.ilabel]
            try:
                d[s].append(a)
            except KeyError:
                d[s] = [a]

    # Indexed by ilabel
    _indexed_arc_cache = {s: {a.ilabel: a.nextstate for a in arcs}
                          for s, arcs
                          in _arcs_cache.items()}

    def simplify_automata_iter(arcs_cache, first_symbol_arcs_cache, substr):
        """Yield shortcut edges (from, to) for a word.

        O(V * |w|) time comes from this function, which we wrap around k
        elements. This is because we are doing a string equality check over
        substr on all edges, though we implement it via BFS.
        """
        first = substr[0]  # Get first character
        # TODO(mkuchnik): Factor out copies
        try:
            first_symbol = inverted_input_symbols[first]
        except KeyError:
            return

        try:
            d = first_symbol_arcs_cache[first_symbol]
        except KeyError:
            return
        # Find all arcs matching on the first string
        # It's possible that the symbol is "a" and all arcs are "a", so we will
        # match on many "a" "aa" "aaa" etc. in a chain.
        # In that case, we would match on all edges, and the longest edge would
        # traverse all |w| length of those edges.
        for s in d.keys():
            curr_state = s
            past_states = [s]
            success = True
            for nth_char in substr:
                try:
                    nth_symbol = inverted_input_symbols[nth_char]
                except KeyError:
                    success = False
                    break
                next_state = _indexed_arc_cache[curr_state].get(
                    nth_symbol, None)
                if next_state is not None:
                    curr_state = next_state
                    past_states.append(next_state)
                else:
                    success = False
                    break
            if success:
                yield s, curr_state, past_states

    one = fst.Weight.one(automata.weight_type())
    new_automata = automata.copy()

    # Iterate over O(k) vocab items
    for word, token in full_vocab.items():
        if len(word) > 1:  # Only simplify non-character tokens
            # Do O(E * |w|) work
            # For every state, we see if an arc matches on a prefix. If it
            # does,
            # we can continue to explore those arcs via BFS until hitting depth
            # |w|.
            # Right now, we have total complexity of O(k * E * |w|), where
            # k * E are usually much bigger than states.
            edge_iter = simplify_automata_iter(_arcs_cache,
                                               _first_symbol_arcs_cache,
                                               word)
            for s_from, s_to, traceback in edge_iter:
                if token_remapper:
                    remapped_token = token_remapper.encode(token)
                else:
                    remapped_token = token
                arc = fst.Arc(remapped_token, remapped_token, one, s_to)
                new_automata.add_arc(s_from, arc)

    if token_remapper:
        inverted_full_vocab = {token_remapper.encode(v): k
                               for k, v in full_vocab.items()}
        relm.automata.attach_symbol_table(new_automata, inverted_full_vocab)
    else:
        inverted_full_vocab = {v: k for k, v in full_vocab.items()}
        relm.automata.attach_symbol_table(new_automata, inverted_full_vocab)

    return new_automata


def _transducer_from_token_list(list_of_tokens, out_token):
    """Create an transducer from a list of tokens.

    The automaton should accept the list.

    For example, [1, 2, 3] will create an automata that accepts "1 2 3".
    The out_token will be used to output on the last edge,
    so "1 2 3" -> out_token.

    :param list list_of_tokens: A python list or tuple containing the sequence
    of tokens to accept
    :param: out_token: A token representing what the list maps to.
    :return: An openfst automata
    """
    # NOTE(mkuchnik): Alternatively, we can use pynini with accep method using
    # the special bracket syntax e.g., "[1][2][3]" which should convert the
    # bracket content to integer.
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    zero = fst.Weight.zero(f.weight_type())
    f.reserve_states(len(list_of_tokens))
    states = []

    # Start State
    start_state = f.add_state()
    f.set_start(start_state)
    states.append(start_state)

    # Transition states
    for i, t in enumerate(list_of_tokens):
        last_state = states[i]
        s = f.add_state()
        f.reserve_arcs(last_state, 1)  # Optional.
        if i == len(list_of_tokens) - 1:
            f.add_arc(last_state, fst.Arc(t, out_token, one, s))
        else:
            f.add_arc(last_state, fst.Arc(t, 0, one, s))
        states.append(s)

    # Let's set all states to have non-accepting final weight
    for s in states:
        f.set_final(s, zero)

    # End states
    end_state = states[-1]
    # By default, state gets infinite weight or zero weight
    # Any state that has non-infinite final weight is a final state.
    f.set_final(end_state, one)

    return f


def build_word_tokenizer_transducer_from_vocabulary(full_vocab):
    """Build a transducer from a word to token vocabulary.

    full_vocab is a dict mapping from string to int.
    """
    transducers = []
    # Build a transducer going from characters to tokens
    # For example: 't' -> 'h' -> 'e' => 'the'
    # Input symbols are thus the characters (ordinal encoding)
    # Output symbols are thus the tokens (full_vocab encoding)
    input_symbols = dict()
    output_symbols = dict()
    for i, (k, v) in enumerate(full_vocab.items()):
        if v == 0:
            # NOTE(mkuchnik): 0 is reserved
            logger.warning("Dropping zero k={}, v={}".format(k, v))
            continue
        k_mapped_toks = []
        for ko in k:
            # Gthe -> G -> t -> h -> e
            km = full_vocab[ko]
            input_symbols[km] = ko
            k_mapped_toks.append(km)
        output_symbols[v] = k
        transducer = _transducer_from_token_list(k_mapped_toks, v)
        transducers.append(transducer)

    input_symbols[0] = "ε"
    output_symbols[0] = "ε"

    tokenizer_transducer = relm.automata.union_automatas(transducers)

    final_state = relm.automata.automata_final_states(tokenizer_transducer)
    start_state = relm.automata.automata_start_state(tokenizer_transducer)
    for fs in final_state:
        tokenizer_transducer.add_arc(fs, fst.Arc(0, 0, 0., start_state))

    tokenizer_transducer = relm.automata.attach_symbol_table(
        tokenizer_transducer, input_symbols, attach_outputs=False)
    tokenizer_transducer = relm.automata.attach_symbol_table(
        tokenizer_transducer, output_symbols, attach_inputs=False)
    if not tokenizer_transducer.verify():
        raise RuntimeError("Automata not valid")
    return tokenizer_transducer


def simplify_automata_symbols_openfst(
        automata, full_vocab,
        token_remapper: Optional[RegexTokenRemapper] = None):
    """Add new edges where an equivalent one can be made with vocab.

    This works by composing the character accepting automata with a transducer
    from characters to word tokens.
    """
    if not automata.verify():
        raise ValueError("Automata not valid")
    if token_remapper:
        full_vocab = {k: token_remapper.encode(v) for (k, v)
                      in full_vocab.items()}
    # NOTE(mkuchnik): Composition requires the first automata's output alphabet
    # to be the same as the second automata's input alphabet. Here, we assume
    # that there is an ordinal encoding.
    tokenizer_transducer = \
        build_word_tokenizer_transducer_from_vocabulary(full_vocab)
    automata_symbols = dict(automata.output_symbols())
    if 0 not in automata_symbols:
        automata_symbols[0] = "ε"
    else:
        assert automata_symbols[0] == "ε"
    tokenizer_symbols = dict(tokenizer_transducer.input_symbols())
    common_symbols = set()
    tokenizer_symbol_values = set(tokenizer_symbols.values())
    for k, v in automata_symbols.items():
        if v in tokenizer_symbol_values:
            common_symbols.add(v)
    used_keys = relm.automata.used_keys_set(automata)
    used_values = {automata_symbols[k] for k in used_keys}
    if not used_values.issubset(common_symbols):
        extra_values = used_values - common_symbols
        raise ValueError("Automata using symbols that are not common: {}."
                         .format(extra_values))
    automata_symbols = {k: v for k, v in automata_symbols.items() if v in
                        common_symbols}
    relm.automata.attach_symbol_table(automata, automata_symbols,
                                      attach_inputs=False)
    if automata_symbols != tokenizer_symbols:
        raise ValueError("Automata output symbols:\n{}\nmismatch tokenizer's "
                         "input symbols:\n{}".format(
                             automata_symbols, tokenizer_symbols))
    automata.arcsort()
    fc = fst.compose(automata, tokenizer_transducer)
    # We are only interested in the output now, convert back to acceptor
    fc = fc.project("output")
    # Remove epsilons and optimize automata
    fc = fc.rmepsilon()
    fc = relm.automata.finalize_automata(fc)
    return fc


def delete_transducer(alphabet):
    """Create a delete transducer of edit distance 1."""
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    epsilon = 0
    f.reserve_states(2)

    start_state = f.add_state()
    f.set_start(start_state)

    end_state = f.add_state()
    f.set_final(end_state, one)

    # Start Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, start_state)
        f.add_arc(start_state, arc)

    # Transitions
    for t in alphabet:
        arc = fst.Arc(t, epsilon, one, end_state)
        f.add_arc(start_state, arc)

    # End Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, end_state)
        f.add_arc(end_state, arc)

    return f


def insert_transducer(alphabet):
    """Create a insert transducer of edit distance 1."""
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    epsilon = 0
    f.reserve_states(2)

    start_state = f.add_state()
    f.set_start(start_state)

    end_state = f.add_state()
    f.set_final(end_state, one)

    # Start Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, start_state)
        f.add_arc(start_state, arc)

    # Transitions
    for t in alphabet:
        arc = fst.Arc(epsilon, t, one, end_state)
        f.add_arc(start_state, arc)

    # End Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, end_state)
        f.add_arc(end_state, arc)

    return f


def substitution_transducer(alphabet):
    """Create a substitution transducer of edit distance 1."""
    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    f.reserve_states(2)

    start_state = f.add_state()
    f.set_start(start_state)

    end_state = f.add_state()
    f.set_final(end_state, one)

    # Start Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, start_state)
        f.add_arc(start_state, arc)

    # Transitions
    for t1 in alphabet:
        for t2 in alphabet:
            if t1 == t2:
                continue
            else:
                arc = fst.Arc(t1, t2, one, end_state)
                f.add_arc(start_state, arc)

    # End Loops
    for t in alphabet:
        arc = fst.Arc(t, t, one, end_state)
        f.add_arc(end_state, arc)

    return f


def levenshtein_transducer(alphabet,
                           passthrough_keys=None,
                           allow_deletes: bool = True,
                           allow_inserts: bool = True,
                           allow_substitutions: bool = True,
                           allow_passthrough_deletes: bool = False,
                           allow_passthrough_substitutions: bool = False):
    """Create a Levenshtein transducer of edit distance 1.

    Inserts, Deletes, and Substitutions over the alphabet are allowed.
    The full edit set is applied to the alphabet. Passthrough keys are
    keys which are ignored for edits, but are still allowed to be accepted
    without edits.

    For example, alphabet can be ASCII characters, but passthrough would be the
    set of characters used in the original automata (e.g., a superset
    of ASCII). The characters that overlapping the two will be editted as
    normal, but the remaining characters will, unless otherwise specified, just
    get mapped linearly through (without edits).
    """
    alphabet = set(alphabet)
    passthrough_keys = set(passthrough_keys)

    f = fst.VectorFst()
    one = fst.Weight.one(f.weight_type())
    epsilon = 0
    f.reserve_states(2)

    # Zero edits state
    start_state = f.add_state()
    f.set_start(start_state)

    # One edit state
    end_state = f.add_state()
    f.set_final(end_state, one)

    if epsilon in alphabet:
        # Epsilon is a metacharacter
        raise ValueError("Found 0 (epsilon) in alphabet")
    if epsilon in passthrough_keys:
        # Epsilon is a metacharacter
        raise ValueError("Found 0 (epsilon) in passthrough keys")

    # Passthrough: These are keys which are not edited, but may be used
    # For example, if alphabet is ASCII, these can be unicode characters we are
    # just letting through, but we don't consider edits using these characters
    # i.e., they can be substituted with ASCII, but Unicode won't be
    # substituted for existing ASCII.
    if passthrough_keys:
        # These are keys that are only in passthrough (not in alphabet)
        remaining_alphabet = passthrough_keys - alphabet
        assert not remaining_alphabet.intersection(alphabet), \
            "Expected alphabet and remaining alphabet to be mutex"
        logger.info("Remaining keys getting passthrough: {}".format(
            remaining_alphabet)
        )
    else:
        remaining_alphabet = set()

    # Start Loops
    for t in alphabet:
        # All characters that are mapped without edits stay at start
        arc = fst.Arc(t, t, one, start_state)
        f.add_arc(start_state, arc)
    for t in remaining_alphabet:
        # All characters that are mapped without edits stay at start
        arc = fst.Arc(t, t, one, start_state)
        f.add_arc(start_state, arc)

    # Delete
    if allow_deletes:
        for t in alphabet:
            # All characters that are deletes (t -> nothing) incur cost
            arc = fst.Arc(t, epsilon, one, end_state)
            f.add_arc(start_state, arc)
    # Passthrough Deletes
    if allow_passthrough_deletes:
        for t in remaining_alphabet:
            # All characters that are deletes (t -> nothing) incur cost
            arc = fst.Arc(t, epsilon, one, end_state)
            f.add_arc(start_state, arc)

    # Insert
    if allow_inserts:
        for t in alphabet:
            # All characters that are nothing (nothing -> t) incur cost
            arc = fst.Arc(epsilon, t, one, end_state)
            f.add_arc(start_state, arc)

    # Substitution
    if allow_substitutions:
        for t1 in alphabet:
            for t2 in alphabet:
                # All characters that are substitutions (t1 -> t2) incur cost
                if t1 != t2:
                    arc = fst.Arc(t1, t2, one, end_state)
                    f.add_arc(start_state, arc)
    # Passthrough Substitutions
    # Here we allow passthrough to be substituted with alphabet
    if allow_passthrough_substitutions:
        for t1 in remaining_alphabet:
            for t2 in alphabet:
                # All characters that are substitutions (t1 -> t2) incur cost
                if t1 != t2:
                    arc = fst.Arc(t1, t2, one, end_state)
                    f.add_arc(start_state, arc)

    # End Loops
    for t in alphabet:
        # All characters that are mapped without edits stay at end
        arc = fst.Arc(t, t, one, end_state)
        f.add_arc(end_state, arc)
    for t in remaining_alphabet:
        # All characters that are mapped without edits stay at end
        arc = fst.Arc(t, t, one, end_state)
        f.add_arc(end_state, arc)

    return f
