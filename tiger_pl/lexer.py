import re
from abc import abstractmethod, ABC
from io import StringIO
from typing import Iterator, Tuple, Union, Optional


"""
This class heirarchy is a bit weird. I _wanted_ a single tree with a root
abstract class named "Token". Under that tree, some token classes are enum-like
(which use a fancy meta class) and some aren't. However Python doesn't let
you mix your own metaclass with ABCMeta, so here we are.


_AbstractToken,Token
    Identifier
    IntegerLiteral
    StringLiteral
_AbstractEnumToken,Token
    Punctuation
    Operator
    Keyword
"""

class _AbstractToken(ABC):
    @classmethod
    @abstractmethod
    def matches(cls, tkn: str) -> bool:
        raise NotImplementedError


class _EnumMeta(type):
    """
    Use the class's attributes as token names and values

    e.g.
    class MyEnumToken(metaclass=_EnumMeta):
        CATS = 'meow'
        DOGS = 'woof'

    assert MyEnumToken._names == ['CATS', 'DOGS']
    assert MyEnumToken._values == ['meow', 'woof']
    """
    def __init__(cls, *args, **kwargs):
        super(_EnumMeta, cls).__init__(*args, **kwargs)
        cls._names = [
            attr_name
            for attr_name in dir(cls)
            if attr_name.upper() == attr_name and not attr_name.startswith("_")
        ]
        cls._items = {attr_name: getattr(cls, attr_name) for attr_name in cls._names}
        cls._values = [getattr(cls, attr_name) for attr_name in cls._names]


class _AbstractEnumToken(metaclass=_EnumMeta):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return tkn in cls._values

    @classmethod
    def values(cls):
        return cls._values


class Token(object):
    def __getattr__(self, attr_name):
        """
        Rather than having to do isinstance(token, SomeToken), this allows
        for token.is_some_token
        """
        if attr_name.startswith("is_"):
            cls_name = attr_name[3:].title().replace("_", "")
            return self.__class__.__name__ == cls_name

        raise AttributeError("{} has no attribute {}".format(self, attr_name))

    def __init__(self, token_value: str):
        if not self.matches(token_value):
            raise ValueError(
                '"{}" is not a valid {}'.format(token_value, self.__class__.__name__)
            )
        self.value = token_value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return type(self) == type(other) and self.value == other.value

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.value)


class Punctuation(Token, _AbstractEnumToken):
    PAREN_OPEN = "("
    PAREN_CLOSE = ")"
    BRACKET_OPEN = "["
    BRACKET_CLOSE = "]"
    CURLY_OPEN = "{"
    CURLY_CLOSE = "}"
    COLON = ":"
    ASSIGNMENT = ":="
    DOT = "."
    COMMA = ","
    SEMI_COLON = ";"


class Operator(Token, _AbstractEnumToken):
    MUL = "*"
    DIV = "/"
    ADD = "+"
    SUB = "-"
    EQ = "="
    NE = "<>"
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    AND = "&"
    OR = "|"


class Keyword(Token, _AbstractEnumToken):
    ARRAY = "array"
    BREAK = "break"
    DO = "do"
    ELSE = "else"
    END = "end"
    FOR = "for"
    FUNCTION = "function"
    IF = "if"
    IN = "in"
    LET = "let"
    NIL = "nil"
    OF = "of"
    THEN = "then"
    TO = "to"
    TYPE = "type"
    VAR = "var"
    WHILE = "while"


class Identifier(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r"[a-zA-Z][a-zA-Z0-9_]*", tkn))


class IntegerLiteral(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return bool(re.match(r"[0-9]+", tkn))


class StringLiteral(Token, _AbstractToken):
    @classmethod
    def matches(cls, tkn: str) -> bool:
        return True


class TokenizerException(Exception):
    pass


class TigerLexer(object):
    def __init__(self, program: str):
        self._program = program

    def next(self) -> Optional[Token]:
        """
        Get next token and advance in token stream.
        If end of stream has been reached, return None.
        """
        raise NotImplementedError("TODO")

    def peek(self, n: int = 0) -> Optional[Token]:
        """
        Peek n tokens ahead without advancing stream.
        """
        raise NotImplementedError("TODO")
