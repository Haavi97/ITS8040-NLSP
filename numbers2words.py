#!/usr/bin/env python3

import pynini as pn
import pytest


@pytest.fixture(scope="module")
def n2w_fst():
    single_digits = pn.string_map({"0": "zéro", "1": "un", "2": "deux", "3": "trois", "4": "quatre",
                                   "5": "cinq", "6": "six", "7": "sept", "8": "huit", "9": "neuf"})
    
    transformer = single_digits

    return transformer


def n2w(fst, number_as_string):
    return (number_as_string * fst).stringify()


@pytest.mark.parametrize("test_input,expected", [
    ("1", "un"),
    ("0", "zero"),
    ("10",  "dix"),
    ("21",  "vingt-et-un"),
    ("10",  "dix"),
    ("30",  "trente"),
    ("21",  "vingt-et-un"),
    ("45",  "quarante-cinq"),
    ("99",  "quatre-vingt-dix-neuf"),
    ("100",  "cent"),
    ("110",  "cent-dix"),
    ("121",  "cent-vingt-et-un"),
    ("100000", "cent-mille"),
    ("0.46", "zero-virgule-quatre-six"),
    ("0.046", "zero-virgule-zero-quatre-six"),
])
def test_numbers(n2w_fst, test_input, expected):
    assert n2w(n2w_fst, test_input) == expected
