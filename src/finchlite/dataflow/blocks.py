from ..finch_assembly.nodes import *

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

class CFGBuilderContext:
    def __init__(self):
        self.blocks = {}
        self.block_counter = 0
        self.current_block = self.new_block()
    
    def new_block(self):
        bid = f"b{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

    def build(self, node: AssemblyNode):
        return self(node)
    
    def __call__(self, node: AssemblyNode):
        match node:
            case Literal(_) | Variable(_, _) | GetAttr(_, _) | Load(_, _) | Store(_, _, _) | Length(_, _) | Slot(_, _) | Stack(_, _):
                return
            case Assign(Variable(name, _), val):
                self.current_block.add_statement(("assign", name, val))
            case Block(bodies):
                for body in bodies:
                    self(body)
            case If(cond, body):
                cond_block = self.current_block
                cond_block.add_statement(("if_cond", cond))
                then_block = self.new_block()
                self.current_block = then_block
                self(body)
                after_block = self.new_block()
                cond_block.add_successor(then_block.id)
                cond_block.add_successor(after_block.id)
                self.current_block.add_successor(after_block.id)
                self.current_block = after_block
            case IfElse(cond, body, else_body):
                cond_block = self.current_block
                cond_block.add_statement(("if_else_cond", cond))
                then_block = self.new_block()
                self.current_block = then_block
                self(body)
                after_block = self.new_block()
                cond_block.add_successor(then_block.id)
                cond_block.add_successor(after_block.id) # i think it's not necessary
                self.current_block.add_successor(after_block.id)

                else_block = self.new_block()
                self.current_block = else_block
                self(else_body)
                self.current_block.add_successor(after_block.id)
                self.current_block = after_block
            case WhileLoop(cond, body):
                cond_block = self.current_block
                cond_block.add_statement(("while_cond", cond))
                body_block = self.new_block()
                after_block = self.new_block()
                cond_block.add_successor(body_block.id)
                cond_block.add_successor(after_block.id)
                self.current_block = body_block
                self(body)
                self.current_block.add_successor(cond_block.id)
                self.current_block = after_block
            case ForLoop(var, start, end, body):
                init_block = self.current_block
                init_block.add_statement(("for_init", var, start, end))

                cond_block = self.new_block()
                init_block.add_successor(cond_block.id)
                cond_block.add_statement(("for_init", var, start, end))

                body_block = self.new_block()
                cond_block.add_successor(body_block)
                
                after_block = self.new_block()
                cond_block.add_successor(after_block)

                self.current_block = body_block
                self(body_block)

                self.current_block.add_statement(("for_inc", var))
                self.current_block.add_successor(cond_block.id)

                self.current_block = after_block
            case Return(value): # TODO: when Return is met probably return None and check if self.current_block == None, if yes go to the end of the function
                self.current_block.add_statement(("return", value))
                self.current_block = self.new_block()
            case BufferLoop(buf, var, body):
                raise NotImplementedError(node) # TODO: change when BufferLoop is implemented (what is a BufferLoop)?
            case Function(Variable(func_name, return_t), args, body): # TODO: create a separe CFG for every function
                ...
            case Break(): # TODO: is it really necessary to handle break when we always proceed
                ...
            case Module(funcs): # TODO: handle modules with multiple functions
                ...
            case node:
                raise NotImplementedError(node)
        
        return self.blocks