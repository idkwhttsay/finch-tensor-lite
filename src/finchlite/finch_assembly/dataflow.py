from abc import abstractmethod

from ..symbolic import DataFlowAnalysis
from ..util import qual_str
from .cfg_builder import assembly_build_cfg
from .nodes import (
    AssemblyNode,
    Assert,
    Assign,
    Call,
    Literal,
    TaggedVariable,
    Variable,
)


def assembly_copy_propagation(node: AssemblyNode):
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


class AbstractAssemblyDataflow(DataFlowAnalysis):
    def stmt_str(self, stmt, state: dict) -> str:
        """Annotate a statement with lattice values."""
        match stmt:
            case Assign(lhs, rhs):
                rhs_str = self._expr_str(rhs, state)
                return f"{str(lhs)} = {rhs_str}"
            case Assert(exp):
                exp_str = self._expr_str(exp, state)
                return f"assert({exp_str})"
            case _:
                return str(stmt)

    def _expr_str(self, expr, state: dict) -> str:
        """Annotate an expression with lattice values."""
        match expr:
            case Literal(value):
                return qual_str(value)
            case TaggedVariable(Variable(name, _), id):
                var_str = f"{name}_{id}"
                if name in state:
                    return f"{var_str}: {self.print_lattice_value(name, state)}"
                return var_str
            case Call(Literal(_) as lit, args):
                annotated_args = [self._expr_str(arg, state) for arg in args]
                return f"{qual_str(lit.val)}({', '.join(annotated_args)})"
            case _:
                return str(expr)

    @abstractmethod
    def print_lattice_value(self, name, state: dict) -> str:
        """Format the lattice value associated with a variable for annotation."""
        ...


class AssemblyCopyPropagation(AbstractAssemblyDataflow):
    def direction(self) -> str:
        """
        Copy propagation is a forward analysis.
        """
        return "forward"

    def transfer(self, stmts, state: dict) -> dict:
        new_state = state.copy()

        for stmt in stmts:
            match stmt:
                case Assign(TaggedVariable(var, _), rhs):
                    # name of the assigned variable
                    var_name = var.name

                    # resolve RHS to its root lattice value
                    resolved_rhs = self._resolve_root(rhs, new_state)
                    new_state[var_name] = resolved_rhs

                    # invalidate any copies that directly point to this variable name
                    to_remove: list[str] = []
                    for name_i, val_i in new_state.items():
                        if self._get_variable_name(val_i) == var_name:
                            to_remove.append(name_i)
                    
                    for name_i in to_remove:
                        new_state.pop(name_i)

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        result = {}

        # only keep copy relationships that exist in both states with the same value
        for var_name in state_1:
            if var_name in state_2 and self._values_equal(
                state_1[var_name], state_2[var_name]
            ):
                result[var_name] = state_1[var_name]

        return result

    def print_lattice_value(self, name, state: dict) -> str:
        value = state.get(name)
        match value:
            case Literal(v):
                return qual_str(v)
            case _:
                return str(value)

    def _get_variable_name(self, var) -> str | None:
        match var:
            case TaggedVariable(Variable(name, _), _):
                return name
            case _:
                return None

    def _values_equal(self, val1, val2) -> bool:
        """Check if two values are equal."""
        if isinstance(val1, TaggedVariable) and isinstance(val2, TaggedVariable):
            return val1 == val2

        if isinstance(val1, Literal) and isinstance(val2, Literal):
            return val1.val == val2.val

        return False

    def _resolve_root(self, value, state: dict):
        current = value
        visited: set[str] = set()

        while True:
            match current:
                case Literal(_):
                    return current
                case TaggedVariable(Variable(name, _), _):
                    # reached an already visited lattice value -> return
                    if name in visited:
                        return current
                    
                    visited.add(name)
                    nxt = state.get(name)
                    
                    # root lattice value found -> return
                    if nxt is None:
                        return current
                    
                    # continue iterating
                    current = nxt
                    continue
                case _:
                    return current
