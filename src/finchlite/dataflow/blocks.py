from ..finch_assembly.nodes import *

# TODO: Do a system design for how to build Basic Blocks and CFG based on the AST. Understand how to handle loops, breaks, continues, function calls, etc. \
# and how to represent them in the CFG. Also, figure out a better way of representiang the connections between blocks (probably a map will work best).

class BasicBlock:
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.successors = []
        self.predecessors = []

    def add_statement(self, statement):
        self.statements.append(statement)

    def add_successor(self, succ_id):
        if succ_id not in self.successors:
            self.successors.append(succ_id)        

    def __repr__(self):
        return f"BasicBlock(id={self.id}, stmts={self.statements}, succs={self.successors})"