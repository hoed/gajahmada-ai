import ast
from dataclasses import dataclass, field
import math
import operator
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class LogicalRule:
    rule_id: str
    antecedents: List[str]
    consequent: str
    description: str = ""


@dataclass
class SymbolicDeductionResult:
    applicable: bool
    math_expression: Optional[str] = None
    math_value: Optional[float] = None
    initial_facts: List[str] = field(default_factory=list)
    derived_facts: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    syllogism_valid: Optional[bool] = None
    conclusion: Optional[str] = None
    confidence: float = 1.0
    explanation: List[str] = field(default_factory=list)


class SymbolicReasoner:
    """
    Deterministic Symbolic & Mathematical Reasoner (Layer 5B).
    Operates 100% independently of any LLM or neural model:
    - Safe AST mathematical & algebraic function evaluation (`sqrt`, `log`, `sin`, `cos`, arithmetic, powers).
    - Forward-chaining propositional & Horn-clause rule deduction (Modus Ponens / Modus Tollens).
    - Categorical syllogism & transitive relation deduction (`A -> B`, `B -> C` => `A -> C`).
    """

    _ALLOWED_BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _ALLOWED_UNARY_OPS = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _ALLOWED_FUNCS: Dict[str, Callable] = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "max": max,
        "min": min,
    }
    _ALLOWED_CONSTS: Dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        self.rules: Dict[str, LogicalRule] = {}
        self._register_default_rules()

    def _register_default_rules(self):
        defaults = [
            LogicalRule(
                "RULE-CRYPTO-01",
                ["has_ed25519_signature", "signature_verified"],
                "cryptographically_authenticated",
                "If a message has an Ed25519 signature and verification passes, it is cryptographically authenticated.",
            ),
            LogicalRule(
                "RULE-CHAIN-01",
                ["cryptographically_authenticated", "pow_hash_valid", "prev_hash_linked"],
                "ledger_block_valid",
                "A block is valid if authenticated, PoW hash satisfies target, and previous hash links.",
            ),
            LogicalRule(
                "RULE-EPISTEMIC-01",
                ["has_primary_source", "no_unresolved_contradictions"],
                "fact_verifiable",
                "A claim with a primary source and no unresolved contradictions is verifiable.",
            ),
            LogicalRule(
                "RULE-SOVEREIGN-01",
                ["runs_offline", "no_mandatory_cloud_api"],
                "sovereign_node_compliant",
                "A node operating offline without mandatory cloud APIs satisfies sovereign architecture compliance.",
            ),
        ]
        for r in defaults:
            self.rules[r.rule_id] = r

    def add_rule(self, rule_id: str, antecedents: List[str], consequent: str, description: str = ""):
        self.rules[rule_id] = LogicalRule(
            rule_id=rule_id,
            antecedents=[a.strip().lower() for a in antecedents],
            consequent=consequent.strip().lower(),
            description=description,
        )

    def evaluate_math(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Extract and deterministically evaluate mathematical expressions in text."""
        cleaned = (
            text.replace("×", "*")
            .replace("÷", "/")
            .replace("^", "**")
        )
        # Allow explicit function calls like sqrt(144) + 5 or arithmetic expressions
        func_pattern = r"((?:sqrt|abs|round|sin|cos|tan|log10|log|exp|ceil|floor)\s*\([^)]+\)(?:\s*[\+\-\*\/\%]{1,2}\s*[0-9\.\(\)a-zA-Z_]+)*)"
        arith_pattern = r"(\(?[0-9]+(?:\.[0-9]+)?\)?(?:\s*(?:\*\*|\/\/|[\+\-\*\/\%])\s*\(?[0-9]+(?:\.[0-9]+)?\)?)+)"

        candidate = None
        m_func = re.search(func_pattern, cleaned)
        if m_func:
            candidate = m_func.group(1).strip()
        else:
            # Avoid matching date strings like 2026-11-22 or ranges like 1945-1967 unless math context or operators
            m_arith = re.search(arith_pattern, cleaned)
            if m_arith:
                expr_str = m_arith.group(1).strip()
                if not re.fullmatch(r"\d{4}\s*-\s*\d{1,4}(?:\s*-\s*\d{1,2})?", expr_str):
                    candidate = expr_str

        if not candidate:
            return None, None

        try:
            tree = ast.parse(candidate, mode="eval")
            val = self._eval_ast(tree.body)
            return candidate, float(val)
        except Exception:
            return None, None

    def _eval_ast(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in self._ALLOWED_CONSTS:
            return self._ALLOWED_CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in self._ALLOWED_BIN_OPS:
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent exceeds safe limit")
            return float(self._ALLOWED_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ALLOWED_UNARY_OPS:
            return float(self._ALLOWED_UNARY_OPS[type(node.op)](self._eval_ast(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if fn_name in self._ALLOWED_FUNCS:
                args = [self._eval_ast(a) for a in node.args]
                return float(self._ALLOWED_FUNCS[fn_name](*args))
        raise ValueError("Unsupported AST node in symbolic math")

    def forward_chain(self, facts: Set[str]) -> Tuple[Set[str], List[str], List[str]]:
        """Run deterministic forward-chaining deduction over known facts and registered rules."""
        known = {f.strip().lower() for f in facts if f.strip()}
        derived: Set[str] = set()
        applied_rules: List[str] = []
        explanations: List[str] = []

        changed = True
        while changed:
            changed = False
            for rule_id, rule in self.rules.items():
                if rule.consequent not in known:
                    if all(ant in known for ant in rule.antecedents):
                        known.add(rule.consequent)
                        derived.add(rule.consequent)
                        applied_rules.append(rule_id)
                        explanations.append(
                            f"[{rule_id}] Because ({' AND '.join(rule.antecedents)}), deduced '{rule.consequent}'."
                        )
                        changed = True
        return derived, applied_rules, explanations

    def evaluate_implications_in_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Parse natural-language or formal implication chains in the prompt:
        - 'If A then B' / 'Jika A maka B'
        - 'All X are Y' / 'Semua X adalah Y'
        and deduce transitive or Modus Ponens conclusions.
        """
        explanations: List[str] = []
        deduced: List[str] = []
        tl = text.strip()

        # Pattern 1: "All A are B. All B are C." or "Semua A adalah B. Semua B adalah C."
        universal_en = re.findall(r"\ball\s+([a-zA-Z0-9_\-]+)\s+are\s+([a-zA-Z0-9_\-]+)", tl, flags=re.IGNORECASE)
        universal_id = re.findall(r"\bsemua\s+([a-zA-Z0-9_\-]+)\s+adalah\s+([a-zA-Z0-9_\-]+)", tl, flags=re.IGNORECASE)
        pairs = [(a.lower(), b.lower()) for a, b in (universal_en + universal_id)]

        # Also check "X is a A" / "X adalah A"
        inst_en = re.findall(r"\b([a-zA-Z0-9_\-]+)\s+is\s+(?:a|an)\s+([a-zA-Z0-9_\-]+)", tl, flags=re.IGNORECASE)
        inst_id = re.findall(r"\b([a-zA-Z0-9_\-]+)\s+adalah\s+([a-zA-Z0-9_\-]+)", tl, flags=re.IGNORECASE)

        if pairs:
            adj: Dict[str, Set[str]] = {}
            for a, b in pairs:
                adj.setdefault(a, set()).add(b)

            # Compute transitive closure
            for start in list(adj.keys()):
                visited = set()
                stack = list(adj.get(start, []))
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        if curr not in adj.get(start, set()):
                            deduced.append(f"All {start} are {curr}")
                            explanations.append(
                                f"[Syllogism Barbara] Transitive universal deduction: All {start} -> {curr}."
                            )
                        for nxt in adj.get(curr, []):
                            if nxt not in visited:
                                stack.append(nxt)
                adj[start].update(visited)

            for inst, cls_name in (inst_en + inst_id):
                inst_l, cls_l = inst.lower(), cls_name.lower()
                if inst_l in ("all", "semua", "what", "who", "apakah"):
                    continue
                if cls_l in adj:
                    for target_cls in sorted(adj[cls_l]):
                        stmt = f"{inst} is {target_cls}"
                        if stmt not in deduced:
                            deduced.append(stmt)
                            explanations.append(
                                f"[Modus Ponens] Since {inst} ∈ {cls_l} and {cls_l} ⊆ {target_cls}, deduced: {stmt}."
                            )

        # Pattern 2: "If P then Q. P is true."
        imp_matches = re.findall(
            r"(?:if|jika)\s+([^,\.]+?)\s+(?:then|maka)\s+([^,\.]+)",
            tl,
            flags=re.IGNORECASE,
        )
        if imp_matches:
            for ant, cons in imp_matches:
                ant_c = ant.strip().lower()
                cons_c = cons.strip().lower()
                # Check if antecedent is asserted in text
                if re.search(rf"\b{re.escape(ant_c)}\b\s+(?:is true|benar|terjadi|holds)", tl, flags=re.IGNORECASE):
                    deduced.append(cons_c)
                    explanations.append(f"[Modus Ponens] Given '{ant_c} -> {cons_c}' and '{ant_c}' holds, deduced '{cons_c}'.")

        return deduced, explanations

    def reason(self, query: str, facts: Optional[List[str]] = None) -> SymbolicDeductionResult:
        math_expr, math_val = self.evaluate_math(query)
        fact_set = set(facts or [])
        derived_rules, applied_rule_ids, rule_explanations = self.forward_chain(fact_set)
        nl_deduced, nl_explanations = self.evaluate_implications_in_text(query)

        all_derived = sorted(derived_rules) + nl_deduced
        all_explanations = list(rule_explanations) + list(nl_explanations)
        if math_expr is not None and math_val is not None:
            formatted = int(math_val) if math_val.is_integer() else round(math_val, 6)
            all_explanations.insert(0, f"[Symbolic Math AST] Evaluated `{math_expr}` = {formatted}")

        applicable = (math_val is not None) or bool(all_derived)
        conclusion_parts = []
        if math_val is not None:
            formatted = int(math_val) if math_val.is_integer() else round(math_val, 6)
            conclusion_parts.append(f"Deterministic mathematical evaluation: `{math_expr} = {formatted}`.")
        if all_derived:
            conclusion_parts.append(f"Symbolic logical deduction: {'; '.join(all_derived)}.")

        return SymbolicDeductionResult(
            applicable=applicable,
            math_expression=math_expr,
            math_value=math_val,
            initial_facts=sorted(fact_set),
            derived_facts=all_derived,
            applied_rules=applied_rule_ids,
            syllogism_valid=True if nl_deduced else None,
            conclusion=" ".join(conclusion_parts) if conclusion_parts else None,
            confidence=1.0 if applicable else 0.0,
            explanation=all_explanations,
        )
