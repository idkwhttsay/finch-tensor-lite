from ..finch_assembly import (
    AssemblyNode,
    Assert,
    Assign,
    Call,
    Literal,
    TaggedVariable,
    Variable,
    assembly_build_cfg,
)
from ..symbolic import DataFlowAnalysis
from ..util import qual_str


def assembly_copy_propagation(node: AssemblyNode):
    ctx = AssemblyCopyPropagation(assembly_build_cfg(node))
    ctx.analyze()
    return ctx


class AssemblyCopyPropagation(DataFlowAnalysis):
    def direction(self) -> str:
        """
        Copy propagation is a forward analysis.
        """
        return "forward"

    def transfer(self, stmts, state: dict) -> dict:
        new_state = state.copy()

        for stmt in stmts:
            match stmt:
                case Assign(lhs, rhs):
                    # Get the variable name being assigned to
                    var_name = self._get_variable_name(lhs)

                    if var_name is not None:
                        new_state[var_name] = rhs

                        # invalidate any copies that point
                        # to the variable being assigned
                        to_remove = []
                        for var, val in new_state.items():
                            if val == lhs:
                                to_remove.append(var)

                        for var in to_remove:
                            new_state.pop(var)

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

    def _get_variable_name(self, var) -> str | None:
        match var:
            case Variable(name, _):
                return name
            case TaggedVariable(Variable(name, _), _):
                return name
            case _:
                return None

    def _values_equal(self, val1, val2) -> bool:
        """
        Check if two values are equal.
        """
        if isinstance(val1, (Variable, TaggedVariable)) and isinstance(
            val2, (Variable, TaggedVariable)
        ):
            return val1 == val2

        if isinstance(val1, Literal) and isinstance(val2, Literal):
            return val1.val == val2.val

        return False

    def __str__(self) -> str:
        """
        Print the CFG with dataflow annotations embedded in statements.
        Annotate variables in expressions with their known values.
        """
        lines = []
        blocks = list(self.cfg.blocks.values())

        for block in blocks:
            # block header
            if block.successors:
                succ_names = [succ.id for succ in block.successors]
                succ_str = f"#succs=[{', '.join(succ_names)}]"
            else:
                succ_str = "#succs=[]"

            lines.append(f"{block.id}: {succ_str}")

            # get the input state for this block
            input_state = self.input_states.get(block.id, {})

            # annotate each statement with analysis output
            for stmt in block.statements:
                annotated_stmt = self._annotate_statement(stmt, input_state)
                lines.append(f"    {annotated_stmt}")

            lines.append("")

        return "\n".join(lines)

    def _annotate_statement(self, stmt, state: dict) -> str:
        """Annotate a statement with dataflow information."""

        match stmt:
            case Assign(lhs, rhs):
                rhs_str = self._annotate_expression(rhs, state)
                return f"{str(lhs)} = {rhs_str}"
            case Assert(exp):
                exp_str = self._annotate_expression(exp, state)
                return f"assert({exp_str})"
            case _:
                return str(stmt)

    def _annotate_expression(self, expr, state: dict) -> str:
        """Recursively annotate an expression with dataflow information."""

        match expr:
            case Literal(value):
                return qual_str(value)
            case TaggedVariable(Variable(name, _), id):
                var_str = f"{name}_{id}"
                if name in state:
                    return f"{var_str}: {str(state[name])}"
                return var_str
            case Variable(name, _):
                if name in state:
                    return f"{name}: {str(state[name])}"
                return str(name)
            case Call(Literal(_) as lit, args):
                # annotate each argument
                annotated_args = [self._annotate_expression(arg, state) for arg in args]
                return f"{qual_str(lit.val)}({', '.join(annotated_args)})"
            case _:
                return str(expr)
