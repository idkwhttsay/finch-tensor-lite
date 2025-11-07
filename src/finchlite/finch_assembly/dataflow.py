from abc import abstractmethod

from ..symbolic import DataFlowAnalysis
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
    """Run copy-propagation on a FinchAssembly node.

    Args:
        node: Root FinchAssembly node to analyze.

    Returns:
        AssemblyCopyPropagation: The completed analysis context.
    """
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


class AbstractAssemblyDataflow(DataFlowAnalysis):
    """Assembly-specific base for dataflow analyses."""

    def stmt_str(self, stmt, state: dict) -> str:
        """Annotate a statement with lattice values."""
        lattice_annotations: list[tuple[str, object]] = []

        match stmt:
            case Assign(_, rhs):
                self.print_lattice_value(lattice_annotations, state, rhs)
                if lattice_annotations:
                    annotations = ", ".join(
                        f"{name} = {str(val)}" for name, val in lattice_annotations
                    )
                    return f"{str(stmt)} \t# {annotations}"

                return str(stmt)
            case Assert(exp):
                self.print_lattice_value(lattice_annotations, state, exp)
                if lattice_annotations:
                    annotations = ", ".join(
                        f"{name} = {str(val)}" for name, val in lattice_annotations
                    )
                    return f"{str(stmt)} \t# {annotations}"

                return str(stmt)
            case _:
                return str(stmt)

    @abstractmethod
    def print_lattice_value(self, annotated_pairs, state, expr) -> str:
        """Format the lattice value associated with the dataflow's lattice."""
        ...


class AssemblyCopyPropagation(AbstractAssemblyDataflow):
    """Copy propagation for FinchAssembly.

    Lattice: a mapping ``{ var_name: TaggedVariable }`` describing simple copy
    relationships between variables. The analysis is forward and only records
    direct copies; expressions and literals are not propagated.
    """

    def direction(self) -> str:
        """
        Copy propagation is a forward analysis.
        """
        return "forward"

    def print_lattice_value(
        self, annotated_pairs: list[tuple[str, object]], state: dict, expr
    ):
        """Collect lattice annotations for variables mentioned in ``expr``."""
        match expr:
            case TaggedVariable(Variable(name, _), id):
                var_str = f"{name}_{id}"
                if name in state:
                    annotated_pairs.append((var_str, state[name]))
                return
            case Call(Literal(_), args):
                for arg in args:
                    self.print_lattice_value(annotated_pairs, state, arg)
                return
            case _:
                return

    def transfer(self, stmts, state: dict) -> dict:
        """Transfer function over a sequence of statements.

        Applies copy-propagation effects of each statement in order, returning
        the updated lattice mapping. Only copies of variables are recorded; any
        existing mappings that point to a variable being reassigned are
        invalidated first.

        Args:
            stmts: Iterable of Assembly statements in a basic block.
            state: Incoming lattice mapping.

        Returns:
            dict: The outgoing lattice mapping after processing ``stmts``.
        """
        new_state = state.copy()

        for stmt in stmts:
            match stmt:
                case Assign(TaggedVariable(var, _), rhs):
                    var_name = var.name

                    # invalidate any copies that directly point to this variable name
                    to_remove: list[str] = []
                    for name, val in new_state.items():
                        if (
                            isinstance(val, TaggedVariable)
                            and val.variable.name == var_name
                        ):
                            to_remove.append(name)

                    for name in to_remove:
                        new_state.pop(name)

                    # after all copies invalidated, add new state for variable `var`
                    root_rhs = self._resolve_root(rhs, new_state)
                    if root_rhs is not None:
                        new_state[var_name] = root_rhs

        return new_state

    def join(self, state_1: dict, state_2: dict) -> dict:
        """Meet operator for copy-propagation.
        (set-like intersection on keys where values are equal).
        """
        result = {}

        # only keep copy relationships that exist in both states with the same value
        for var_name in state_1:
            if var_name in state_2 and state_1[var_name] == state_2[var_name]:
                result[var_name] = state_1[var_name]

        return result

    def _resolve_root(self, value, state: dict):
        """Follow copy links to the ultimate TaggedVariable.

        Starting from ``value``, chase simple copies through ``state`` until a
        non-copied value is reached. Returns the final TaggedVariable if the
        chain stays within variables; otherwise returns ``None`` for expressions
        and literals.
        """
        current = value
        visited: set[str] = set()

        while True:
            match current:
                case TaggedVariable(Variable(name, _), _):
                    # reached an already visited variable
                    if name in visited:
                        return current

                    visited.add(name)
                    nxt = state.get(name)

                    # root lattice value found
                    # or next lattice value is not TaggedVariable
                    if nxt is None or not isinstance(nxt, TaggedVariable):
                        return current

                    # continue iterating
                    current = nxt
                    continue
                case _:
                    return None
