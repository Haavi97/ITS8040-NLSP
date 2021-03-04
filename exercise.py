#!/usr/bin/env python3

import pynini as pn
fst = (pn.a("a") | pn.a("e")) + pn.t("a", pn.a("0").closure(0,5)) | pn.t(pn.a("a").star, "0") + pn.a("xxx")

"""\b((a|e)a)|(a*xxx)\b"""

def top10_paths(fst):
    return list(pn.shortestpath(fst, nshortest=10).paths().ostrings())

# only one output
print(top10_paths('xxx'*fst))
print(top10_paths('axxx'*fst))
print(top10_paths('aaxxx'*fst))
# jne. So the regex:  a*xxx

print(top10_paths('aa'*fst)) # More than 2 outputs
print(top10_paths('ea'*fst)) # More than 2 outputs

