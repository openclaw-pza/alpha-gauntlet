#!/usr/bin/env python3
# Ported from QuantaAlpha (https://github.com/QuantaAlpha/QuantaAlpha), MIT License, Copyright (c) Ziyi Tang et al.
"""factor DSL — factor-expression grammar parsing (str -> AST) + complexity metrics + max-common-subtree similarity.

Ported from QuantaAlpha's quantaalpha/factors/coder/factor_ast.py (MIT License). This file is a streamlined
port aimed at reskin pre-screening of crypto volume-price factors; semantics align with the original, and it
neither pulls in QuantaAlpha dependencies nor executes the original repo's code.

—— Points aligned with the original ——
1. Grammar: pyparsing.infix_notation defines 7 operator-precedence levels (unary minus > mul/div > add/sub >
   comparison > logical-and > logical-or > ternary); lexing: variable = Optional('$') + identifier, numbers
   support scientific notation.
2. AST nodes: VarNode / NumberNode / FunctionNode / BinaryOpNode / ConditionalNode / UnaryOpNode (dataclasses).
3. Complexity metrics: symbol length (string length), base-feature count ($-prefixed unique variables),
   free-arg count (NumberNode count), unique-var count, total AST node count.
4. Max common subtree: preorder-enumerate every subtree root, double-traverse to find the largest subtree of
   equal size that is equal after commutative normalization. Commutative operators (+ * == != & && | ||) and
   commutative functions (MIN/MAX/ADD/MULTIPLY) sort their children for normalization, so reordered
   expressions are judged isomorphic.

—— Local customizations (not in the original) ——
- Field set aligned to this system: o/h/l/c/v/vwap/returns (both $-prefixed and bare forms).
- similarity ∈ [0,1]: max_common_subtree_size / min(size1, size2), i.e. containment similarity. One
  expression being a substructure of another (including reskin reorder) -> 1.0; no common structure -> 0.0.

Read-only reference discipline: this module modifies no existing module; only reskin_screen.py / tests call it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Tuple
from typing import Optional as Opt

from pyparsing import (
    Combine,
    DelimitedList,
    Forward,
    Literal,
    Optional,
    ParserElement,
    Regex,
    Word,
    alphanums,
    alphas,
    infix_notation,
    one_of,
    opAssoc,
)

# packrat speeds up nested parsing; raise the recursion limit for deep expressions.
ParserElement.enable_packrat()
sys.setrecursionlimit(5000)


# --------------------------------------------------------------------------- #
# AST nodes (ports factor_ast.py L17-126 semantics)
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """AST base class."""

    def __str__(self) -> str:  # pragma: no cover - subclasses override
        return "<Node>"


@dataclass
class VarNode(Node):
    """Variable node: a base feature like $close, or a bare identifier."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class NumberNode(Node):
    """Numeric constant node."""

    value: float

    def __str__(self) -> str:
        # Integer values drop the decimal point for a cleaner round-trip string (24.0 -> "24").
        if self.value == int(self.value) and abs(self.value) < 1e15:
            return str(int(self.value))
        return repr(self.value)


@dataclass
class FunctionNode(Node):
    """Function-call node. args holds all actual arguments."""

    name: str
    args: List[Node] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


@dataclass
class BinaryOpNode(Node):
    """Binary operator node."""

    op: str
    left: Node
    right: Node

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class ConditionalNode(Node):
    """Ternary conditional node: cond ? a : b."""

    condition: Node
    true_expr: Node
    false_expr: Node

    def __str__(self) -> str:
        return f"({self.condition} ? {self.true_expr} : {self.false_expr})"


@dataclass
class UnaryOpNode(Node):
    """Unary operator node, e.g. -x."""

    op: str
    operand: Node

    def __str__(self) -> str:
        return f"({self.op}{self.operand})"


# --------------------------------------------------------------------------- #
# Grammar (ports factor_ast.py L128-237 semantics, with pyparsing 3.x snake_case API)
# --------------------------------------------------------------------------- #
# Lexing
_var = Combine(Optional(Literal("$")) + Word(alphas, alphanums + "_"))
_number = Regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")

# Operators
_mul_div = one_of("* /")
_add_sub = one_of("+ -")
_comparison = one_of("> < >= <= == !=")
_logical_and = one_of("&& &")
_logical_or = one_of("|| |")
_conditional = ("?", ":")


def _make_var(tokens):
    return VarNode(tokens[0])


def _make_number(tokens):
    return NumberNode(float(tokens[0]))


def _unwrap(arg):
    """Collapse a pyparsing intermediate into a single Node (ports factor_ast unwrap semantics)."""
    if isinstance(arg, Node):
        return arg
    # ParseResults / list: take its first Node.
    try:
        seq = list(arg)
    except TypeError:
        return arg
    if len(seq) == 1:
        return _unwrap(seq[0])
    # With multiple elements, find the first Node (a function-name var node appears first, stripped by the caller).
    for el in seq:
        u = _unwrap(el)
        if isinstance(u, Node):
            return u
    return _unwrap(seq[0])


def _make_function(tokens):
    # tokens: [funcNameVarNode, '(', arg1, arg2, ..., ')']
    name_tok = tokens[0]
    name = name_tok.name if isinstance(name_tok, VarNode) else str(name_tok)
    raw_args = list(tokens[2:-1])
    args = [_unwrap(a) for a in raw_args]
    assert all(isinstance(a, Node) for a in args), f"Invalid function args: {args}"
    return FunctionNode(name, args)


def _make_binary(tokens):
    t = tokens[0]
    if len(t) == 3:
        return BinaryOpNode(t[1], _unwrap(t[0]), _unwrap(t[2]))
    # Left-associative chain: a op b op c -> ((a op b) op c)
    result = _unwrap(t[0])
    for i in range(1, len(t) - 1, 2):
        result = BinaryOpNode(t[i], result, _unwrap(t[i + 1]))
    return result


def _make_conditional(tokens):
    t = tokens[0]
    return ConditionalNode(_unwrap(t[0]), _unwrap(t[2]), _unwrap(t[4]))


def _make_unary(tokens):
    t = tokens[0]
    return UnaryOpNode(t[0], _unwrap(t[1]))


# Expression grammar
_expr = Forward()
_var.set_parse_action(_make_var)
_number.set_parse_action(_make_number)

_function_call = _var + "(" + Optional(DelimitedList(_expr)) + ")"
_function_call.set_parse_action(_make_function)

_operand = (
    _function_call
    | _var
    | _number
    | ("(" + _expr + ")").set_parse_action(lambda t: t[1])
)

_unary_minus = Literal("-")

_expr <<= infix_notation(
    _operand,
    [
        (_unary_minus, 1, opAssoc.RIGHT, _make_unary),
        (_mul_div, 2, opAssoc.LEFT, _make_binary),
        (_add_sub, 2, opAssoc.LEFT, _make_binary),
        (_comparison, 2, opAssoc.LEFT, _make_binary),
        (_logical_and, 2, opAssoc.LEFT, _make_binary),
        (_logical_or, 2, opAssoc.LEFT, _make_binary),
        (_conditional, 3, opAssoc.RIGHT, _make_conditional),
    ],
)


def parse(text: str) -> Node:
    """Parse an expression string -> AST root node. Raises ValueError on failure."""
    try:
        result = _expr.parse_string(text, parse_all=True)
        return result[0]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Failed to parse expression {text!r}: {e}") from e


# Alias compatible with the original's parse_expression name (ports factor_ast.py L239).
parse_expression = parse


# --------------------------------------------------------------------------- #
# This system's field set (not in the original; used for field-name validation)
# --------------------------------------------------------------------------- #
# Accepts both $-prefixed and bare forms. returns and ret are aliases (the incumbent registry uses $ret).
BASE_FEATURES = {"o", "h", "l", "c", "v", "vwap", "returns", "ret"}
BASE_FEATURE_ALIASES = {
    "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v",
}


def is_base_feature(name: str) -> bool:
    """Whether name is a base feature (after stripping $, it lands in the field set or alias set)."""
    bare = name[1:] if name.startswith("$") else name
    return bare in BASE_FEATURES or bare in BASE_FEATURE_ALIASES


# --------------------------------------------------------------------------- #
# Complexity metrics (ports factor_ast.py L387-560 semantics)
# --------------------------------------------------------------------------- #
def _walk(node: Node):
    """Preorder-traverse all nodes (including node itself)."""
    yield node
    if isinstance(node, FunctionNode):
        for a in node.args:
            yield from _walk(a)
    elif isinstance(node, BinaryOpNode):
        yield from _walk(node.left)
        yield from _walk(node.right)
    elif isinstance(node, ConditionalNode):
        yield from _walk(node.condition)
        yield from _walk(node.true_expr)
        yield from _walk(node.false_expr)
    elif isinstance(node, UnaryOpNode):
        yield from _walk(node.operand)


def symbol_length(expr: str) -> int:
    """Symbol length = expression string length (a proxy for structural complexity). Ports calculate_symbol_length."""
    return len(expr.strip())


def count_free_args(expr_or_node) -> int:
    """free-arg count = NumberNode count (number of tunable free parameters). Ports count_free_args."""
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    return sum(1 for n in _walk(node) if isinstance(n, NumberNode))


def count_unique_vars(expr_or_node) -> int:
    """unique-var count = number of $-prefixed unique variable names. Ports count_unique_vars."""
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    names = {n.name for n in _walk(node) if isinstance(n, VarNode) and n.name.startswith("$")}
    return len(names)


def count_base_features(expr_or_node) -> int:
    """base-feature count = size of the $-prefixed unique base-feature set (ER(f,h) usage penalty).

    Ports count_base_features. The original semantics are "$-prefixed unique variable count"; this
    implementation keeps it consistent (the same set as count_unique_vars).
    """
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    names = {n.name for n in _walk(node) if isinstance(n, VarNode) and n.name.startswith("$")}
    return len(names)


def count_all_nodes(expr_or_node) -> int:
    """Total AST node count (operators/functions/variables/constants). Ports count_all_nodes / count_nodes."""
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    return sum(1 for _ in _walk(node))


def free_arg_ratio(expr_or_node) -> float:
    """free-arg ratio = free-arg count / total AST node count. Denser constants mean a higher-dimensional search space."""
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    total = count_all_nodes(node)
    if total == 0:
        return 0.0
    return count_free_args(node) / total


def unique_var_ratio(expr_or_node) -> float:
    """unique-var ratio = unique-var count / total AST node count. Variable diversity density."""
    node = expr_or_node if isinstance(expr_or_node, Node) else parse(expr_or_node)
    total = count_all_nodes(node)
    if total == 0:
        return 0.0
    return count_unique_vars(node) / total


def complexity(expr: str) -> dict:
    """Summarize all complexity metrics of one expression, for CLI/report use."""
    node = parse(expr)
    return {
        "symbol_length": symbol_length(expr),
        "n_nodes": count_all_nodes(node),
        "n_base_features": count_base_features(node),
        "n_free_args": count_free_args(node),
        "n_unique_vars": count_unique_vars(node),
        "free_arg_ratio": free_arg_ratio(node),
        "unique_var_ratio": unique_var_ratio(node),
    }


# --------------------------------------------------------------------------- #
# Max common subtree similarity (ports factor_ast.py L278-383 semantics + commutative normalization)
# --------------------------------------------------------------------------- #
# Commutative binary operators (ports is_commutative_op, L310-312).
_COMMUTATIVE_BINOPS = {"+", "*", "==", "!=", "&", "&&", "|", "||"}
# Commutative functions (local enhancement: MIN/MAX/ADD/MULTIPLY argument order is irrelevant).
_COMMUTATIVE_FUNCS = {"MIN", "MAX", "ADD", "MULTIPLY"}
# Function-form <-> binary-operator-form synonym normalization (local enhancement, to prevent reskin evasion):
# MULTIPLY(a,b) ≡ a*b, ADD(a,b) ≡ a+b, SUBTRACT(a,b) ≡ a-b, DIVIDE(a,b) ≡ a/b.
# An incumbent using MULTIPLY(-1, x) and an imitator rewriting it as -1*x should be judged isomorphic
# (previously they would fall below threshold and be missed). Only effective for exactly-2-arg function
# calls; synonym binary operators are unified to the canonical symbol on the dict's right.
_FUNC_TO_BINOP = {"ADD": "+", "SUBTRACT": "-", "MULTIPLY": "*", "DIVIDE": "/"}


@dataclass
class SubtreeMatch:
    """Max common subtree match result."""

    root1: Node
    root2: Node
    size: int


def subtree_size(node: Node) -> int:
    """Subtree node count (ports get_subtree_size)."""
    return sum(1 for _ in _walk(node))


def _all_subtrees(node: Node) -> List[Node]:
    """Preorder-enumerate all subtree roots (ports get_all_subtrees)."""
    return list(_walk(node))


def _canon_key(node: Node) -> str:
    """Commutative-normalized canonical key of a node: serialize after sorting children of commutative operators/functions.

    Makes a+b equal to b+a, MIN(x,y) equal to MIN(y,x), generating the same key, used for the commutative
    matching of are_subtrees_equal (replacing the original L324-330 two-ordering attempt, generalized to n-ary
    commutative functions).
    """
    if isinstance(node, VarNode):
        return f"V:{node.name}"
    if isinstance(node, NumberNode):
        return f"N:{node.value!r}"
    if isinstance(node, UnaryOpNode):
        return f"U:{node.op}:{_canon_key(node.operand)}"
    if isinstance(node, BinaryOpNode):
        return _binop_key(node.op, _canon_key(node.left), _canon_key(node.right))
    if isinstance(node, ConditionalNode):
        return (
            f"C:{_canon_key(node.condition)}:"
            f"{_canon_key(node.true_expr)}:{_canon_key(node.false_expr)}"
        )
    if isinstance(node, FunctionNode):
        # 2-arg arithmetic functions (MULTIPLY/ADD/SUBTRACT/DIVIDE) normalize to the equivalent binary-operator
        # key, so MULTIPLY(-1,x) and -1*x are judged isomorphic (closing the reskin-evasion gap).
        if node.name in _FUNC_TO_BINOP and len(node.args) == 2:
            return _binop_key(
                _FUNC_TO_BINOP[node.name],
                _canon_key(node.args[0]),
                _canon_key(node.args[1]),
            )
        keys = [_canon_key(a) for a in node.args]
        if node.name in _COMMUTATIVE_FUNCS:
            keys = sorted(keys)
        return f"F:{node.name}:({','.join(keys)})"
    return f"?:{node!r}"


def _binop_key(op: str, lk: str, rk: str) -> str:
    """Binary-operator canonical key: serialize after sorting a commutative operator's two operand keys.

    Factored out for reuse, so a BinaryOpNode and a normalized 2-arg arithmetic FunctionNode use the same key,
    guaranteeing a*b and MULTIPLY(a,b), a+b and ADD(a,b) generate the same canon_key.
    """
    if op in _COMMUTATIVE_BINOPS:
        lk, rk = sorted((lk, rk))
    return f"B:{op}:{lk}:{rk}"


def subtrees_equal(n1: Node, n2: Node) -> bool:
    """Recursively compare whether two subtrees are structurally equal; commutative operator/function children sorted then compared.

    Ports are_subtrees_equal (L314-338), generalizing the original's binary two-ordering attempt to canonical-key comparison.
    """
    return _canon_key(n1) == _canon_key(n2)


def find_largest_common_subtree(root1: Node, root2: Node) -> Opt[SubtreeMatch]:
    """Max common subtree of two ASTs (ports find_largest_common_subtree, L278-360).

    Implementation optimization: index root2's subtrees by (size, canon_key), avoiding the original's O(n1*n2)
    full-pairing deep comparison; the result is equivalent — still returns the size of the largest structurally-equal subtree.
    """
    subs1 = _all_subtrees(root1)
    subs2 = _all_subtrees(root2)

    # root2 subtrees: canon_key -> a subtree root with that structure (any will do).
    key2root = {}
    for st in subs2:
        k = _canon_key(st)
        if k not in key2root:
            key2root[k] = st

    max_match: Opt[SubtreeMatch] = None
    max_size = 0
    for st1 in subs1:
        size1 = subtree_size(st1)
        if size1 <= max_size:
            continue
        k1 = _canon_key(st1)
        st2 = key2root.get(k1)
        if st2 is not None:
            max_size = size1
            max_match = SubtreeMatch(st1, st2, size1)
    return max_match


def similarity(expr1: str, expr2: str) -> float:
    """Reskin similarity of two expressions ∈ [0,1].

    Definition: max common subtree size / min(tree1 size, tree2 size) — containment similarity.
    - One expression being entirely a substructure of another (including commutative reorder) -> 1.0
    - No common structure -> 0.0
    The decision is not made here: this score is only an early warning; truth/falsity is judged by the downstream IC gauntlet.
    """
    t1 = parse(expr1)
    t2 = parse(expr2)
    return _similarity_nodes(t1, t2)


def _similarity_nodes(t1: Node, t2: Node) -> float:
    s1 = subtree_size(t1)
    s2 = subtree_size(t2)
    denom = min(s1, s2)
    if denom == 0:
        return 0.0
    match = find_largest_common_subtree(t1, t2)
    common = match.size if match is not None else 0
    return common / denom


def similarity_detail(expr1: str, expr2: str) -> Tuple[float, Opt[SubtreeMatch], int, int]:
    """Return (similarity, SubtreeMatch, size1, size2), for reports to display the matched subtree."""
    t1 = parse(expr1)
    t2 = parse(expr2)
    s1, s2 = subtree_size(t1), subtree_size(t2)
    match = find_largest_common_subtree(t1, t2)
    common = match.size if match is not None else 0
    denom = min(s1, s2) or 1
    return common / denom, match, s1, s2


if __name__ == "__main__":  # pragma: no cover - manual smoke
    samples = [
        "DELTA($close, 24) / $close",
        "MULTIPLY(-1, DELTA($close, 6) / $close)",
        "ATR($high, $low, $close, 14) / ($close + 1e-12) * 100",
    ]
    for s in samples:
        node = parse(s)
        print(f"{s}")
        print(f"  repr : {node}")
        print(f"  cx   : {complexity(s)}")
    print("sim(mom24, mom24-reskin):",
          similarity("DELTA($close, 24) / $close", "DELTA($c, 24) / $c"))
