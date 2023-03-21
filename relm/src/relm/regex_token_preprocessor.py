"""Define preprocessors for used on regex automata."""

import string
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pywrapfst as fst

import relm.relm_logging
from relm.regex_token_remapper import RegexTokenRemapper

logger = relm.relm_logging.get_logger()


automata_type = fst.VectorFst
optional_remapper = Optional[RegexTokenRemapper]


class AutomataPreprocessor(ABC):
    """Define a wrapper class around a automata preprocessing phase."""

    @abstractmethod
    def transform(self, automata: automata_type,
                  token_remapper: optional_remapper = None) -> automata_type:
        """Process the automata and return a new one."""
        raise NotImplementedError("Transform is not implemented.")


class LevenshteinTransformer(AutomataPreprocessor):
    """Performs edit distance 1 transduction on the automata."""

    def __init__(self, symbol_set: Optional[str] = None, num_edits:
                 Optional[int] = None,
                 allow_deletes: bool = True,
                 allow_inserts: bool = True,
                 allow_substitutions: bool = True,
                 allow_passthrough_deletes: bool = False,
                 allow_passthrough_substitutions: bool = False):
        """
        Initialize the transformer with a symbol set.

        If a symbol set is given, use that. Otherwise, use all printables.
        """
        if symbol_set is None:
            # NOTE(mkuchnik): We don't include the full whitespace set
            symbol_set = (string.ascii_letters + string.digits +
                          string.punctuation + " ")
        if num_edits is None:
            num_edits = 1
        if not isinstance(symbol_set, str):
            raise ValueError("Symbol set should be a string. Got {}.".format(
                type(symbol_set)))
        if not isinstance(num_edits, int):
            raise ValueError("Num edits should be a int. Got {}.".format(
                type(num_edits)))
        self.symbol_set = symbol_set
        self.num_edits = num_edits
        self.allow_deletes = allow_deletes
        self.allow_inserts = allow_inserts
        self.allow_substitutions = allow_substitutions
        self.allow_passthrough_deletes = allow_passthrough_deletes
        self.allow_passthrough_substitutions = allow_passthrough_substitutions

    def transform(self, automata: automata_type,
                  token_remapper: optional_remapper = None,
                  union_automatas_kwargs=None) -> automata_type:
        """Process the automata and return a new one."""
        # TODO(mkuchnik): We make strong assumptions here e.g., ASCII input
        logger.info(
            "Running edit distance {} transformation on the automata with "
            "symbols: {}.".format(self.num_edits, self.symbol_set)
        )
        automata, edit_automata = self._get_edit_automata(
            automata, token_remapper)
        new_automatas = [automata]
        new_automata = automata
        for _ in range(self.num_edits):
            new_automata = fst.compose(new_automata, edit_automata)
            new_automata = new_automata.project("output")
            new_automatas.append(new_automata)
        if not union_automatas_kwargs:
            union_automatas_kwargs = {}
        new_automata = relm.automata.union_automatas(
            new_automatas, **union_automatas_kwargs)
        return new_automata

    def _get_edit_automata(self, automata: automata_type,
                           token_remapper: optional_remapper = None
                           ) -> Tuple[automata_type, automata_type]:
        """Return the automata used for processing and the new automata.

        Edits operate over the integer values of the character set and are
        applied in a preprocessing pass. In the simple case that the charset of
        both edits is a superset of the automata, the edits are the standard
        interprettation. Otherwise, if there are characters in the automata
        that are not in the charset, then we, by default, just pass them
        through without any edits.
        """
        symbol_set = set([x for x in self.symbol_set])
        if token_remapper:
            symbol_table = {token_remapper.encode(ord(x)): x
                            for x in symbol_set}
        else:
            symbol_table = {ord(x): x for x in symbol_set}
        symbol_table[0] = 'ε'
        # Materialize edit automata using only symbol table
        charset = list(symbol_table.keys())
        charset = set(charset)
        if 0 in charset:
            # Epsilon is a metacharacter
            charset.remove(0)
        # Then, join on foreign keys to get joined symbol tables
        foreign_symbols = dict(automata.output_symbols())
        used_keys = relm.automata.used_keys_set(automata)
        missing_keys = used_keys - charset
        if 0 in missing_keys:
            # Epsilon is a metacharacter
            missing_keys.remove(0)
        # Add missing keys as a passthrough
        allow_passthrough_substitutions = self.allow_passthrough_substitutions
        if missing_keys:
            logger.warning("Missing Levenshtein charset used in automata:\n{}."
                           " Used keys: {}. Charset: {}. Automata: {}"
                           .format(missing_keys, used_keys, charset,
                                   relm.automata.summarize_automata(automata),
                                   ))
        edit_automata = relm.regex_graph_optimizations.levenshtein_transducer(
            charset,
            passthrough_keys=missing_keys,
            allow_deletes=self.allow_deletes,
            allow_inserts=self.allow_inserts,
            allow_substitutions=self.allow_substitutions,
            allow_passthrough_deletes=self.allow_passthrough_deletes,
            allow_passthrough_substitutions=allow_passthrough_substitutions,
        )
        for k, v in foreign_symbols.items():
            if k in symbol_table and symbol_table[k] != v:
                logger.warning("Overwriting symbol {}={} with {}".format(
                    k, symbol_table[k], v))
            assert isinstance(k, int), "Expected int, got {}".format(k)
            assert isinstance(v, str), "Expected str, got {}".format(v)
            symbol_table[k] = v
        edit_automata.arcsort(sort_type="ilabel")
        # NOTE(mkuchnik): We assume automata is already in remapped space
        relm.automata.attach_symbol_table(automata, symbol_table)
        relm.automata.attach_symbol_table(edit_automata, symbol_table)
        return automata, edit_automata
