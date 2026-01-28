import numpy as np
import pandas as pd
import ast


def extract_element(t):
    if type(t) != list:
        return [t]
    xs = []
    for e in t:
        xs += extract_element(e)
    return xs


def stringTest(code):
    ast = parser.st2list(ast.parse(code))
    leaves = extract_element(ast)
    return "while" not in leaves and "for" not in leaves


def test(result, ans):
    np.testing.assert_allclose(result, ans)
    return 1
