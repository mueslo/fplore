# -*- coding: utf-8 -*-
from pyparsing import (Word, Literal, Regex, Keyword, CaselessKeyword, Forward,
                       Group, OneOrMore, ZeroOrMore, alphas, nums, Suppress,
                       delimitedList, CharsNotIn, Empty, Optional, Or,
                       restOfLine)
from fractions import Fraction
from orderedattrdict import AttrDict


IDENTIFIER = Word(alphas + "_", alphas + nums + "_")
INT_DECIMAL = Regex(r'([+-]?(([1-9][0-9]*)|0+))')
INTEGER = INT_DECIMAL.setParseAction(
    lambda s, l, t: int(t[0]))
FLOAT = Regex(r'[+-]?((((\d+\.\d*)|(\d*\.\d+))'
              r'([eE][-+]?\d+)?)|(\d*[eE][+-]?\d+)|INF)').setParseAction(
    lambda s, l, t: float(t[0]))
FLAG = Regex(r'(?P<key>[A-Z_]+)\((?P<val>[+-])\)').setParseAction(
    lambda s, l, t: (t.key, t.val == '+'))
(LPAREN, RPAREN, LBRACK, RBRACK, LBRACE, RBRACE,
 SEMI, COMMA, EQUAL, DQUOTE) = map(Suppress, "()[]{};,=\"")
SIZE = (delimitedList(INTEGER | IDENTIFIER) | INTEGER | Literal('*') |
        IDENTIFIER)
STRING = DQUOTE + ZeroOrMore(CharsNotIn('"')) + DQUOTE
FRACTION = (INTEGER + Literal('/') + INTEGER).setParseAction(
    lambda s, l, t: Fraction(t[0], t[2]))
BOOLEAN = (CaselessKeyword("t") | CaselessKeyword("f")).setParseAction(
    lambda s, l, t: t[0].lower() == "t")

VALUE = Forward()
SVALUE = FLOAT | FRACTION | INTEGER | STRING | BOOLEAN | FLAG
VALUE << (SVALUE | (LBRACE + Group(delimitedList(VALUE) | Empty()
                                   ).setParseAction(
    lambda s, l, t: t.asList()) + RBRACE))
VARIABLE = (IDENTIFIER("name") + Optional(LBRACK + SIZE("size") + RBRACK))
SCALARTYPE = Or(map(Keyword, "int real logical flag char string".split()))
STRUCTMEMBERS = Forward()
STRUCTTYPE = Keyword("struct") + LBRACE + STRUCTMEMBERS("members") + RBRACE
DECLARATION = ((SCALARTYPE | STRUCTTYPE)("type") +
               Optional(LBRACK + SIZE + RBRACK) + VARIABLE)
STRUCTMEMBERS << Group(ZeroOrMore(Group(DECLARATION + SEMI)))
DECL_ASS_STMT = DECLARATION + Optional(EQUAL + VALUE("value")) + SEMI

SECTION = (Keyword('section') + IDENTIFIER('name') + LBRACE +
           Group(OneOrMore(Group(DECL_ASS_STMT)))('declarations') +
           RBRACE + SEMI)
COMMENT = '#' + restOfLine  # todo: could potentially match '#' within strings?
CONFIG = Group(OneOrMore(Group(SECTION)))
CONFIG.ignore(COMMENT)


def walk(ns, declaration, value):
    # walks through declaration tokens recursively and constructs namespace
    if declaration.type == 'struct':
        if declaration.size:
            subdecs_vals = []
            for i, v in enumerate(value):
                fake_declaration = declaration.copy()
                fake_declaration.name = i
                fake_declaration.size = 0  # prevents infinite recursion
                subdecs_vals.append((fake_declaration, v))

            ns[declaration.name] = [None] * len(value)
        else:
            ns[declaration.name] = AttrDict()
            subdecs_vals = zip(declaration.members, value)

        for subdec, val in subdecs_vals:
            for d in walk(ns[declaration.name], subdec, val):
                yield d
    else:
        yield ns, declaration.name, value


class FPLOConfig(object):
    load_default = True

    def _load(self):
        with open(self.filepath, 'r') as config_file:
            config_str = config_file.read()
        self.parse_config(config_str)

    def parse_config(self, config_str):
        tokens = CONFIG.parseString(config_str, parseAll=True)

        self._namespace = AttrDict()
        for config in tokens:
            for section in config:
                section_ns = self._namespace[section.name] = AttrDict()
                for declaration in section.declarations:
                    for nsnode, name, value in walk(
                            section_ns, declaration, declaration.value):
                        nsnode[name] = value

    def __getattr__(self, name):
        return self._namespace[name]
