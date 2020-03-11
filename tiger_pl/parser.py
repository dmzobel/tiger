import inspect
import inspect
from abc import ABC
from typing import List, Union, Optional, Callable

from tiger_interpreter.lexer import (
    Token,
    Keyword,
    TigerLexer,
    Punctuation,
    Operator as OpToken,
)

"""
# Grammar Reference

program → exp
dec → tyDec | varDec | funDec
tyDec → type tyId = ty
ty → tyId | arrTy | recTy
arrTy → array of tyId
recTy → { fieldDec∗, }
fieldDec → id : tyId
funDec → function id ( fieldDec∗, ) = exp
        | function id ( fieldDec∗, ) : tyId = exp
varDec → var id := exn
        | var id : tyId := exp
lValue → id | subscript | fieldExp
subscript → lValue [ exp ]
fieldExp → lValue . id
exp → lValue | nil | intLit | stringLit
    | seqExp | negation | callExp | infixExp
    | arrCreate | recCreate | assignment
    | ifThenElse | ifThen | whileExp | forExp
    | break | letExp
seqExp → ( exp∗; )
negation → - exp
callExp → id ( exp∗, )
infixExp → exp infixOp exp
arrCreate → tyId [ exp ] of exp
recCreate → tyId { fieldCreate∗, }
fieldCreate → id = exp
assignment → lValue := exp
ifthenelse → if exp then exp else exp
ifthen → if exp then exp
whileExp → while exp do exp
forExp → for id := exp to exp do exp
letExp → let dec+ in exp∗; end
"""


"""
# Class Heirarchy for AST Nodes
- Every class (except for Python-abstract ones) references a single rule in
  the grammar
- The Python-abstract classes are just a way to organize related rules.
  That is, they're an implementation decision but not required by the language.

AbstractSyntaxTreeNode
    AbstractIdentifier
        Identifier
        TypeIdentifier
    Expression
        IntegerLiteralExpression
        StringLiteralExpression
        LetExpression
        IfThenExpression
        IfThenElseExpression
        WhileExpression
        ForExpression
        BreakExpression
        NilExpNode
        NegationExpression
        SeqExpression
        AbstractLValueExpression
            LValueExpression
            SubscriptExpression
            FieldExpression
        AssignmentExpression
        ArrCreateExpression
        RecCreateExpression
        CallExpression
    FieldCreate
    Declaration
        TypeDeclaration
        VarDeclaration
        FieldDeclaration
        FunDeclaration
    TigerType
    AbstractTigerType
        ArrayType
        RecType
    Operator
"""


class ParserException(Exception):
    pass


class AbstractSyntaxTreeNode(ABC):
    """
    Both "abstract" in the syntax tree sense and the python sense :)
    """

    def __repr__(self):
        """
        Visualizes instances of this class like
        ClassName(this_arg, that_kwarg='some_default')
        """
        params = inspect.signature(self.__init__).parameters
        args = [
            repr(getattr(self, param.name))
            for param in params.values()
            if param.default == inspect._empty
        ]
        kwargs = [
            "{}={}".format(param.name, repr(getattr(self, param.name, param.default)))
            for param in params.values()
            if param.default != inspect._empty
        ]
        return "{}({}{})".format(
            self.__class__.__name__,
            ", ".join(args),
            ", " + ", ".join(kwargs) if kwargs else "",
        )

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return all(
            getattr(self, p.name, None) == getattr(other, p.name, None)
            for p in inspect.signature(self.__init__).parameters.values()
        )

    def __init__(self, *args, **kwargs):
        """
        A generic equivalent of
        def __init__(self, arg1, arg2=True):
            self.arg1 = arg1
            self.arg2 = arg2

        Basically saves some boilerplate in subclasses by using super().
        Also enforces the init signature as the source of truth.
        """
        params = list(inspect.signature(self.__init__).parameters.values())
        for arg, param in zip(args, params):
            setattr(self, param.name, arg)

        for param in params[len(args) :]:
            kwarg = kwargs.get(param.name, param.default)
            setattr(self, param.name, kwarg)


class AbstractIdentifier(AbstractSyntaxTreeNode, ABC):
    pass


class Declaration(AbstractSyntaxTreeNode):
    pass


class Expression(AbstractSyntaxTreeNode):
    pass


class Identifier(AbstractIdentifier):
    def __init__(self, value: str):
        super().__init__(value)


class TypeIdentifier(AbstractIdentifier):
    def __init__(self, value: str):
        super().__init__(value)


class VarDeclaration(Declaration):
    def __init__(
        self,
        identifier: Identifier,
        expression: Expression,
        type_identifier: Optional[TypeIdentifier] = None,
    ):
        super().__init__(identifier, expression, type_identifier=type_identifier)


class FieldDeclaration(Declaration):
    def __init__(self, identifier: Identifier, type_identifier: TypeIdentifier):
        super().__init__(identifier, type_identifier)


class FunDeclaration(Declaration):
    def __init__(
        self,
        identifier: Identifier,
        field_declarations: List[FieldDeclaration],
        expression: Expression,
        type_identifier: Optional[TypeIdentifier] = None,
    ):
        super().__init__(
            identifier, field_declarations, expression, type_identifier=type_identifier
        )


class AbstractTigerType(AbstractSyntaxTreeNode, ABC):
    pass


class ArrayType(AbstractTigerType):
    def __init__(self, type_identifier: TypeIdentifier):
        super().__init__(type_identifier)


class RecType(AbstractTigerType):
    def __init__(self, field_declarations: List[FieldDeclaration]):
        super().__init__(field_declarations)


class TigerType(AbstractSyntaxTreeNode):
    def __init__(self, tiger_type: Union[TypeIdentifier, ArrayType, RecType]):
        super().__init__(tiger_type)


class TypeDeclaration(Declaration):
    def __init__(self, type_identifier: TypeIdentifier, tiger_type: TigerType):
        super().__init__(type_identifier, tiger_type)


class Operator(AbstractSyntaxTreeNode):
    def __init__(self, op: OpToken):
        super().__init__(op)


class IntegerLiteralExpression(Expression):
    def __init__(self, value: int):
        super().__init__(value)


class StringLiteralExpression(Expression):
    def __init__(self, value: str):
        super().__init__(value)


class LetExpression(Expression):
    def __init__(self, declarations: List[Declaration], expressions: List[Expression]):
        if len(declarations) == 0:
            raise ParserException("Let expressions require at least one declaration.")
        super().__init__(declarations, expressions)


class IfThenExpression(Expression):
    def __init__(self, condition: Expression, consequent: Expression):
        super().__init__(condition, consequent)


class IfThenElseExpression(Expression):
    def __init__(
        self, condition: Expression, consequent: Expression, alternative: Expression
    ):
        super().__init__(condition, consequent, alternative)


class WhileExpression(Expression):
    def __ini__(
        self, condition: Expression, consequent: Expression,
    ):
        super().__init__(condition, consequent)


class ForExpression(Expression):
    def __ini__(
        self,
        identfier: Identifier,
        lower_bound: Expression,
        upper_bound: Expression,
        body: Expression,
    ):
        super().__init__(identfier, lower_bound, upper_bound, body)


class BreakExpression(Expression):
    pass


class NilExpression(Expression):
    pass


class NegationExpression(Expression):
    def __init__(self, expression):
        super().__init__(expression)


class SeqExpression(Expression):
    def __init__(self, expressions):
        super().__init__(expressions)


class AbstractLValueExpression(Expression, ABC):
    pass


class LValueExpression(AbstractLValueExpression):
    def __init__(
        self, l_value: Union[Identifier, "SubscriptExpression", "FieldExpression"],
    ):
        super().__init__(l_value)


class SubscriptExpression(AbstractLValueExpression):
    def __init__(
        self, l_value: LValueExpression, index_expression: Expression,
    ):
        super().__init__(l_value, index_expression)


class FieldExpression(AbstractLValueExpression):
    def __init__(self, l_value: LValueExpression, identifer: Identifier):
        super().__init__(l_value, identifer)


class AssignmentExpression(Expression):
    def __init__(self, l_value: LValueExpression, expression: Expression):
        super().__init__(l_value, expression)


class ArrCreateExpression(Expression):
    def __init__(
        self,
        type_identifier: Identifier,
        size_expression: Expression,
        initial_values_expression: Expression,
    ):
        super().__init__(type_identifier, size_expression, initial_values_expression)


class FieldCreate(AbstractSyntaxTreeNode):
    def __init__(self, identifier: Identifier, expression: Expression):
        super().__init__(identifier, expression)


class RecCreateExpression(Expression):
    def __init__(
        self, type_identifier: Identifier, field_create_expressions: List[FieldCreate],
    ):
        super().__init__(type_identifier, field_create_expressions)


class CallExpression(Expression):
    def __init__(self, identifier: Identifier, expressions: List[Expression]):
        super().__init__(identifier, expressions)


class InfixExpression(Expression):
    def __init__(
        self,
        left_expression: Expression,
        operator: Operator,
        right_expression: Expression,
    ):
        super().__init__(left_expression, operator, right_expression)


class Program(AbstractSyntaxTreeNode):
    """
    For use as the root node of the AST
    """

    def __init__(self, expression: Expression):
        super().__init__(expression)


def _assert_tkn_val(token: Token, expect_val: Union[str, List[str]]):
    if isinstance(expect_val, list):
        if token not in expect_val:
            raise ParserException(
                'Expected one of {}, found "{}"'.format(expect_val, token.value)
            )
    else:
        if token != expect_val:
            raise ParserException(
                'Expected "{}", found "{}"'.format(expect_val, token.value)
            )


def _assert_identifier(token: Token):
    if not token.is_identifier:
        raise ParserException("Expected identifier, found {}".format(token))


class TigerParser(object):
    def parse(self, lexer: TigerLexer) -> Program:
        """
        Consume the tokens from the lexer, building the AST along the way.
        """
        next_token = lexer.peek()
        if next_token == Keyword.LET:
            return Program(self._parse_let_expression(lexer))
        else:
            # TODO: support programs not wrapped in a let
            raise NotImplementedError("TODO")

    def _parse_expression(self, lexer: TigerLexer) -> Expression:
        """
        This method I left implemented, since it calls out to so many others. Plus,
        while it's not super interesting, it does demonstrate the basic idea.

        exp → lValue | nil | intLit | stringLit
            | seqExp | negation | callExp | infixExp
            | arrCreate | recCreate | assignment
            | ifThenElse | ifThen | whileExp | forExp
            | break | letExp
        """
        next_token = lexer.peek()
        if next_token.is_integer_literal:
            exp = IntegerLiteralExpression(lexer.next().value)
        elif next_token.is_string_literal:
            exp = StringLiteralExpression(lexer.next().value)
        elif next_token == Keyword.NIL:
            exp = self._parse_nil_expression(lexer)
        elif next_token == Keyword.BREAK:
            exp = self._parse_break_expression(lexer)
        elif next_token == Keyword.IF:
            exp = self._parse_if_expression(lexer)
        elif next_token == Keyword.WHILE:
            exp = self._parse_while_expression(lexer)
        elif next_token == Keyword.FOR:
            exp = self._parse_for_expression(lexer)
        elif next_token == Keyword.LET:
            exp = self._parse_let_expression(lexer)
        elif next_token == OpToken.SUB:
            exp = self._parse_negation_expression(lexer)
        elif next_token == Punctuation.PAREN_OPEN:
            exp = self._parse_seq_expression(lexer)
        elif next_token.is_identifier:
            next_next = lexer.peek(1)
            if next_next == Punctuation.DOT:
                exp = self._parse_new_field_expression(lexer)
            elif next_next == Punctuation.BRACKET_OPEN:
                exp = self._parse_arr_create_or_subscript_expression(lexer)
            elif next_next == Punctuation.PAREN_OPEN:
                exp = self._parse_call_expression(lexer)
            elif next_next == Punctuation.CURLY_OPEN:
                exp = self._parse_rec_create_expression(lexer)
            else:
                raise ParserException("Unexpected token {}".format(next_token.value))
        else:
            raise ParserException("Unexpected token {}".format(next_token.value))

        if lexer.peek() in OpToken.values():
            op_token = lexer.next()
            right_exp = self._parse_expression(lexer)
            return InfixExpression(exp, Operator(op_token), right_exp,)
        else:
            return exp

    def _parse_let_expression(self, lexer: TigerLexer) -> LetExpression:
        """
        letExp → let dec+ in exp∗; end
        """
        raise NotImplementedError("TODO")

    def _parse_if_expression(
        self, lexer: TigerLexer
    ) -> Union[IfThenExpression, IfThenElseExpression]:
        """
        ifthenelse → if exp then exp else exp
        ifthen → if exp then exp
        """
        raise NotImplementedError("TODO")

    def _parse_while_expression(self, lexer: TigerLexer) -> WhileExpression:
        """
        whileExp → while exp do exp
        """
        raise NotImplementedError("TODO")

    def _parse_for_expression(self, lexer: TigerLexer) -> ForExpression:
        """
        forExp → for id := exp to exp do exp
        """
        raise NotImplementedError("TODO")

    def _parse_break_expression(self, lexer: TigerLexer) -> BreakExpression:
        """
        break
        """
        raise NotImplementedError("TODO")

    def _parse_nil_expression(self, lexer: TigerLexer) -> NilExpression:
        """
        nil
        """
        raise NotImplementedError("TODO")

    def _parse_negation_expression(self, lexer: TigerLexer) -> NegationExpression:
        """
        negation → - exp
        """
        raise NotImplementedError("TODO")

    def _parse_seq_expression(self, lexer: TigerLexer) -> SeqExpression:
        """
        seqExp → ( exp∗; )
        """
        raise NotImplementedError("TODO")

    def _parse_arr_create_or_subscript_expression(
        self, lexer: TigerLexer,
    ) -> Union[LValueExpression, ArrCreateExpression, AssignmentExpression]:
        """
        arrCreate → tyId [ exp ] of exp
        subscript → lValue [ exp ]
        """
        raise NotImplementedError("TODO")

    def _parse_new_field_expression(
        self, lexer: TigerLexer
    ) -> Union[LValueExpression, AssignmentExpression]:
        """
        fieldExp → lValue . id
        """
        raise NotImplementedError("TODO")

    def _parse_rest_of_l_value_expression(
        self, l_value: LValueExpression, lexer: TigerLexer
    ) -> Union[LValueExpression, AssignmentExpression]:
        """
        lValue → id | subscript | fieldExp
        subscript → lValue [ exp ]
        fieldExp → lValue . id

        The base case for l-values is an identifier, which is also the
        left-most token in the expression. This makes it pretty pretty easy to
        spot a new l-value starting (`a.`, `a[`), but then it may nest
        indefinitely (e.g. `a.b[c].d`).

        The calling method (_parse_new_field_expression, 
        _parse_arr_create_or_subscript_expression) is responsible for the
        left-most l-value. This method is for finishing it off.
        """
        raise NotImplementedError("TODO")

    def _parse_assignment_expression(
        self, l_value: LValueExpression, lexer: TigerLexer,
    ) -> AssignmentExpression:
        """
        assignment → lValue := exp
        """
        raise NotImplementedError("TODO")

    def _parse_call_expression(self, lexer: TigerLexer) -> CallExpression:
        """
        callExp → id ( exp∗, )
        """
        raise NotImplementedError("TODO")

    def _parse_rec_create_expression(self, lexer: TigerLexer,) -> RecCreateExpression:
        """
        recCreate → tyId { fieldCreate∗, }
        """
        raise NotImplementedError("TODO")

    def _parse_field_create(self, lexer: TigerLexer) -> FieldCreate:
        """
        fieldCreate → id = exp
        """
        raise NotImplementedError("TODO")

    def _parse_type_declaration(self, lexer: TigerLexer) -> TypeDeclaration:
        """
        tyDec → type tyId = ty
        """
        raise NotImplementedError("TODO")

    def _parse_array_type(self, lexer: TigerLexer) -> ArrayType:
        """
        arrTy → array of tyId
        """
        raise NotImplementedError("TODO")

    def _parse_record_type(self, lexer: TigerLexer) -> RecType:
        """
        recTy → { fieldDec∗, }
        """
        raise NotImplementedError("TODO")

    def _parse_func_declaration(self, lexer: TigerLexer) -> FunDeclaration:
        """
        funDec → function id ( fieldDec∗, ) = exp
                | function id ( fieldDec∗, ) : tyId = exp
        """
        raise NotImplementedError("TODO")

    def _parse_var_declaration(self, lexer: TigerLexer) -> VarDeclaration:
        """
        varDec → var id := exn
                | var id : tyId := exp
        """
        raise NotImplementedError("TODO")

    def _parse_field_declaration(self, lexer: TigerLexer) -> FieldDeclaration:
        """
        fieldDec → id : tyId
        """
        raise NotImplementedError("TODO")
