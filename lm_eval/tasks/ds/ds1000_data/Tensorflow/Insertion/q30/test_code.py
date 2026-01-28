import tensorflow as tf

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
    return "tf" in leaves


def test(result, ans=None):
    try:
        assert result == ans
        return 1
    except:
        return 0
