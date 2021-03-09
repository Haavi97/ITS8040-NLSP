#!/usr/bin/env python3

import pynini as pn
import pytest


@pytest.fixture(scope="module")
def n2w_fst():
    n_2_9 = pn.u(*"23456789")
    numbers = pn.u(*"01", n_2_9)
    numbers2 = pn.closure(numbers, 2)
    numbers3 = pn.closure(numbers, 3)
    sigma_star = (numbers | pn.a("^") | pn.a(" ")).star
    chars = pn.u(*"asdfghjklqwertyuiopzxcvbnm- ^")
    sigma_star_extended = (chars.star | numbers).star
    splitter3 = pn.cdrewrite(pn.t("", "^^^ "), numbers,
                             numbers3, sigma_star, direction='rtl').optimize()
    splitter2 = pn.cdrewrite(pn.t("", "^^ "), numbers,
                             numbers2, sigma_star, direction='rtl').optimize()
    splitter1 = pn.cdrewrite(pn.t("", "^ "), numbers,
                             numbers, sigma_star, direction='rtl').optimize()
    splitter0 = splitter3*splitter2*splitter1

    t_2_to_9 = pn.string_map({" 0": "", "0": "", "2": "deux", "3": "trois", "4": "quatre",
                                    "5": "cinq", "6": "six", "7": "sept", "8": "huit", "9": "neuf"})
    t_1_to_9 = pn.u(pn.string_map({"1": "un"}), t_2_to_9)
    t_0_to_9 = pn.u(pn.string_map({"1": ""}), t_2_to_9 + pn.t("", " "))

    single_digits = pn.u(pn.string_map({"0": "zero", "1": "un"}), t_2_to_9)
    zero = pn.string_map({"0": "zero"})

    # final_one = pn.cdrewrite(pn.t("1", "et-un"), " ","[EOS]", sigma_star_extended)

    tenths = pn.string_map({"2^": "vingt", "3^": "trente", "4^": "quarante",
                            "5^": "cinquante", "6^": "soixante", "7^": "soixante-dix",
                            "8^": "quatre-vingts", "9^": "quatre-vingt-dix", "0^": ""})

    t_10_to_19 = pn.string_map({"1^ 0": "dix", "1^ 1": "onze", "1^ 2": "douze", "1^ 3": "treize",
                                "1^ 4": "quatorze", "1^ 5": "quinze", "1^ 6": "seize"})

    tenths_e = t_10_to_19.ques + \
        (tenths + pn.t(" ", " et ") + t_1_to_9).ques + t_1_to_9.ques

    hundreds = t_0_to_9 + pn.string_map({"^^": "cent"}) + pn.t(" ", " ")

    thousands = hundreds.ques + \
        (tenths_e + pn.t("", " ")).ques + \
        pn.string_map({"^^^": "mille"}) + pn.t(" ", " ")

    final_digits = pn.cdrewrite(
        single_digits, "[BOS]", "[EOS]", sigma_star_extended)

    delete_zeros = pn.cdrewrite(pn.string_map(
        {"0^^^ ": "^^^ ", "0^^ ": "", "0^ ": "", }), sigma_star_extended, sigma_star_extended, sigma_star_extended)

    ws_remover = pn.cdrewrite(
        pn.t(" ", "-"), sigma_star_extended, chars.plus, sigma_star_extended)
    # ws_remover = ws_remover*pn.cdrewrite(pn.t("--", "-"), sigma_star_extended, chars.plus, sigma_star_extended)
    transformer = zero | splitter0*delete_zeros*(thousands.ques + hundreds.ques +
                                                 t_10_to_19.ques + tenths_e.ques)*ws_remover

    return transformer.optimize()


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
