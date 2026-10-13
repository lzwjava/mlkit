"""Agent harnesses: ReAct and Chain-of-Thought loops over a tool set.

Implements the *loop structure* from the reference table —
Thought → Action → Observation → … → Answer (ReAct) and step-by-step
decomposition (CoT) — with a template-based policy and real tools
(calculator, unit conversion, knowledge lookup, python expressions).

The policy is deliberately deterministic: it shows exactly how an agent
harness works independent of which model drives it (swap `policy` for an
LLM call and the loop is unchanged).
"""

from __future__ import annotations

import ast
import operator as _op
import re

# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.Pow: _op.pow, ast.Mod: _op.mod,
    ast.FloorDiv: _op.floordiv, ast.USub: _op.neg, ast.UAdd: _op.pos,
}


def _safe_eval(expr: str):
    """Evaluate an arithmetic expression safely (AST whitelist)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")
    return ev(ast.parse(expr, mode="eval"))


UNIT_TABLE = {
    # unit -> factor to the dimension's base unit
    "length": {"km": 1000.0, "m": 1.0, "cm": 0.01, "mm": 0.001,
               "mi": 1609.344, "ft": 0.3048, "in": 0.0254, "yd": 0.9144},
    "mass": {"kg": 1.0, "g": 0.001, "t": 1000.0, "lb": 0.45359237,
             "oz": 0.028349523},
    "time": {"h": 3600.0, "min": 60.0, "s": 1.0, "ms": 0.001},
}

KNOWLEDGE_BASE = {
    "photosynthesis": "Photosynthesis is the process by which plants convert "
                      "light energy into chemical energy stored in glucose.",
    "gravity": "Gravity is the attractive force between masses; on Earth "
               "objects accelerate at ~9.81 m/s^2.",
    "transformer": "A Transformer is a neural network architecture built on "
                   "self-attention, introduced in 'Attention Is All You Need' "
                   "(2017).",
    "lora": "LoRA freezes pretrained weights and injects trainable low-rank "
            "matrices (A: d×r, B: r×d) into attention projections.",
    "rlhf": "RLHF aligns language models using human preference data via a "
            "reward model and policy optimisation (PPO).",
}


def tool_calculator(arg: str) -> str:
    try:
        val = _safe_eval(arg.strip())
        if isinstance(val, float) and abs(val - round(val)) < 1e-12:
            val = int(round(val))
        return f"calc({arg}) = {val}"
    except Exception as exc:                                   # noqa: BLE001
        return f"calculator error: {exc}"


def tool_convert(arg: str) -> str:
    m = re.match(r"\s*(?:convert\s+)?(-?[\d.]+)\s*([a-zA-Z°]+)\s*"
                 r"(?:to|in)\s*([a-zA-Z°]+)\s*$", arg)
    if not m:
        return "convert error: expected '<value> <unit> to <unit>'"
    val, src, dst = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
    if src in ("c", "celsius") and dst in ("f", "fahrenheit"):
        return f"{val}°C = {val * 9 / 5 + 32:.2f}°F"
    if src in ("f", "fahrenheit") and dst in ("c", "celsius"):
        return f"{val}°F = {(val - 32) * 5 / 9:.2f}°C"
    if src in ("c", "celsius") and dst in ("k", "kelvin"):
        return f"{val}°C = {val + 273.15:.2f}K"
    if src in ("k", "kelvin") and dst in ("c", "celsius"):
        return f"{val}K = {val - 273.15:.2f}°C"
    for table in UNIT_TABLE.values():
        if src in table and dst in table:
            return f"{val} {src} = {val * table[src] / table[dst]:.6g} {dst}"
    return f"convert error: unknown units {src}->{dst}"


def tool_search(arg: str) -> str:
    arg = arg.lower().strip()
    for key, text in KNOWLEDGE_BASE.items():
        if key in arg or any(w in text.lower() for w in arg.split() if len(w) > 4):
            return text
    return f"search: no results for '{arg}'"


def tool_python(arg: str) -> str:
    """Evaluate a python *expression* restricted to safe builtins."""
    import math
    safe = {"abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "len": len, "pow": pow, "int": int, "float": float,
            "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
            "log10": math.log10, "exp": math.exp, "pi": math.pi,
            "e": math.e, "floor": math.floor, "ceil": math.ceil}
    try:
        return repr(eval(arg, {"__builtins__": {}}, safe))     # noqa: S307
    except Exception as exc:                                   # noqa: BLE001
        return f"python error: {exc}"


TOOLS = {
    "calculator": tool_calculator,
    "convert": tool_convert,
    "search": tool_search,
    "python": tool_python,
}


# --------------------------------------------------------------------------
# policy: routes a question to a tool (stand-in for an LLM tool-choice step)
# --------------------------------------------------------------------------

def policy(question: str) -> tuple[str, str] | None:
    """Return (tool_name, argument) or None when no tool applies."""
    q = question.strip()
    if re.search(r"\b(to|in)\b\s*(km|m|mi|ft|cm|kg|g|lb|t|c|f|k|celsius|"
                 r"fahrenheit|kelvin)\b", q, re.I) and \
            re.search(r"[-\d]", q):
        return "convert", q
    if re.search(r"[\d)]\s*[-+*/^%]\s*[\d(]", q) or re.search(
            r"\b(what is|calculate|compute|evaluate)\b", q, re.I):
        expr = re.sub(r"[^\d+\-*/(). %^]|(?<!\d)\.(?!\d)", "", q)
        expr = expr.replace("^", "**").strip()
        if expr:
            return "calculator", expr
    if re.search(r"\b(define|what is|who is|explain|how does)\b", q, re.I):
        return "search", q
    return None


# --------------------------------------------------------------------------
# ReAct / CoT runners
# --------------------------------------------------------------------------

def react(question: str, max_steps: int = 4, verbose: bool = True) -> dict:
    """ReAct loop: Thought -> Action -> Observation ... -> Answer."""
    trace = []
    observation = None
    for step in range(max_steps):
        choice = policy(question) if observation is None else None
        if observation is None:
            if choice is None:
                thought = "no tool needed; answer directly from the question"
                answer = question
                trace.append(("Thought", thought))
                trace.append(("Answer", answer))
                break
            tool, arg = choice
            thought = (f"the question needs {tool}; extract argument "
                       f"'{arg}'")
            trace.append(("Thought", thought))
            trace.append(("Action", f"{tool}[{arg}]"))
            observation = TOOLS[tool](arg)
            trace.append(("Observation", observation))
        else:
            trace.append(("Thought", "observation answers the question"))
            answer = observation
            trace.append(("Answer", answer))
            break
    else:
        answer = observation or "no answer found within step budget"
        trace.append(("Answer", answer))

    if verbose:
        for role, content in trace:
            print(f"    {role:12s}: {content}")
    return {"trace": trace, "answer": answer}


def chain_of_thought(question: str, verbose: bool = True) -> dict:
    """Decompose into sub-steps, solve each, then aggregate."""
    steps = []
    choice = policy(question)
    if choice and choice[0] == "convert":
        steps.append(("parse", f"value/unit extracted from: '{question}'"))
        out = TOOLS["convert"](choice[1])
        steps.append(("compute", out))
        answer = out
    elif choice and choice[0] == "calculator":
        expr = choice[1]
        steps.append(("extract", f"expression: {expr}"))
        steps.append(("intermediate", _show_terms(expr)))
        out = TOOLS["calculator"](expr)
        steps.append(("compute", out))
        answer = out
    else:
        steps.append(("recall", f"lookup for: {question}"))
        out = TOOLS["search"](question)
        steps.append(("retrieve", out))
        answer = out

    if verbose:
        for i, (label, content) in enumerate(steps, 1):
            print(f"    step {i} [{label:12s}]: {content}")
        print(f"    final answer : {answer}")
    return {"steps": steps, "answer": answer}


def _show_terms(expr: str) -> str:
    """Human-readable evaluation order for the arithmetic expression."""
    try:
        import math
        tokens = re.findall(r"\d+\.?\d*|[-+*/()]", expr)
        return "evaluate left-to-right with precedence: " + " ".join(tokens)
    except Exception:                                           # noqa: BLE001
        return expr