from ..finch_assembly.nodes import *
from typing import Dict, List, Any

# TODO: move the entire thing to the finch_assenbly folder (create a cfg.py)

class BasicBlock:
    def __init__(self, id):
        self.id = id
        self.statements = []
        self.successors = []
        self.predecessors = []

    def add_statement(self, statement):
        self.statements.append(statement)

    def add_successor(self, succ_id: str, blocks: Dict[str, 'BasicBlock']) -> None:
        if succ_id not in self.successors:
            self.successors.append(succ_id)
        
        if self.id not in blocks[succ_id].predecessors:
            blocks[succ_id].predecessors.append(self.id)

    def __repr__(self):
        return f"BasicBlock(id={self.id}, stmts={self.statements}, succs={self.successors})"
    
class CFG:
    def __init__(self, func_name: str): # TODO: also pass argument types to allow fuctions with same names but different signatures
        self.name = func_name
        self.blocks = {}
        self.current_block = self.new_block()
        self.block_counter = 0
    
    def new_block(self):
        bid = f"{self.name}_{self.block_counter}"
        self.block_counter += 1
        block = BasicBlock(bid)
        self.blocks[bid] = block
        return block

class CFGBuilderContext: # is it required to have a 'main' entry point for a program?
    def __init__(self):
        self.cfgs = {}
        self.current_cfg = None

    def new_cfg(self, name: str):
        new_cfg = CFG(name)
        self.cfgs[name] = new_cfg
        return new_cfg

    def build(self, node: AssemblyNode):
        return self(node)
    
    def __call__(self, node: AssemblyNode):
        match node:
            # what are Repack, GetAttr, SetAttr, etc.?
            case Literal(_) | Unpack(_, _) | Repack(_) | Resize(_, _) | Variable(_, _) | GetAttr(_, _) | SetAttr(_, _, _) | Load(_, _) | Store(_, _, _) | Length(_) | Slot(_, _) | Stack(_, _):
                return None #TODO: add them as statements
            case Assign(Variable(name, _), val):
                self.current_cfg.current_block.add_statement(("assign", name, val))
            case Block(bodies):
                for body in bodies:
                    self(body)
            case If(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_cond", cond))
                then_block = self.current_cfg.new_block()
                self.current_cfg.current_block = then_block
                self(body)
                after_block = self.current_cfg.new_block()
                cond_block.add_successor(then_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks)
                self.current_cfg.current_block.add_successor(after_block.id, self.current_cfg.blocks)
                self.current_cfg.current_block = after_block
            case IfElse(cond, body, else_body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("if_else_cond", cond))

                then_block = self.current_cfg.new_block()
                self.current_cfg.current_block = then_block
                self(body)
                
                after_block = self.current_cfg.new_block()
                cond_block.add_successor(then_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks) # i think it's not necessary
                self.current_cfg.current_block.add_successor(after_block.id, self.current_cfg.blocks)

                else_block = self.current_cfg.new_block()
                self.current_cfg.current_block = else_block
                self(else_body)
                self.current_cfg.current_block.add_successor(after_block.id, self.current_cfg.blocks)
                self.current_cfg.current_block = after_block
            case WhileLoop(cond, body):
                cond_block = self.current_cfg.current_block
                cond_block.add_statement(("while_cond", cond))

                body_block = self.current_cfg.new_block()
                after_block = self.current_cfg.new_block()
                
                cond_block.add_successor(body_block.id, self.current_cfg.blocks)
                cond_block.add_successor(after_block.id, self.current_cfg.blocks)
                
                self.current_cfg.current_block = body_block
                self(body)
                
                self.current_cfg.current_block.add_successor(cond_block.id, self.current_cfg.blocks)
                self.current_cfg.current_block = after_block
            case ForLoop(var, start, end, body):
                init_block = self.current_cfg.current_block
                init_block.add_statement(("for_init", var, start, end))

                cond_block = self.current_cfg.new_block()
                init_block.add_successor(cond_block.id, self.current_cfg.blocks)
                cond_block.add_statement(("for_init", var, start, end))

                body_block = self.current_cfg.new_block()
                cond_block.add_successor(body_block, self.current_cfg.blocks)
                
                after_block = self.current_cfg.new_block()
                cond_block.add_successor(after_block, self.current_cfg.blocks)

                self.current_cfg.current_block = body_block
                self(body_block)

                self.current_cfg.current_block.add_statement(("for_inc", var))
                self.current_cfg.current_block.add_successor(cond_block.id, self.current_cfg.blocks)

                self.current_cfg.current_block = after_block
            case Return(value): # TODO: do the same thing as in Break but exit to the 'End' block of a function
                self.current_cfg.current_block.add_statement(("return", value))
                self.current_cfg.current_block = self.current_cfg.new_block()
            case BufferLoop(buf, var, body):
                raise NotImplementedError(node) # TODO: implement BufferLoop printing and then implement this
            case Function(Variable(_, _), args, body):
                for arg in args:
                    match arg:
                        case Variable(name, t):
                            self.current_cfg.current_block.add_statement(("func_arg"), name)
                        case _:
                            raise NotImplementedError(f"Unrecognized argument type: {arg}")
                
                self(body)
            case Call(Literal(_) as lit, args): # do we have to the handle it here value when it's handled in 'Assign'?
                ...
            case Break():
                # TODO: pass the exit node whenever diving into the for/while and exit to that block (look at the interpreter)
                ...
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(f"Unrecognized function type: {type(func)}")
                    
                    self.current_cfg = self.new_cfg(func.name.name)
                    self(func)
                    end_func_block = self.current_cfg.new_block()
                    self.current_cfg.current_block.add_successor(end_func_block, self.current_cfg.blocks)
                    self.current_cfg.current_block = end_func_block
                return None
            case node:
                raise NotImplementedError(node)
        
        return self.cfgs


"""
cfgs: {
    main: {
        main_0: {...statements},
        main_1: {...statements},
        main_2: {...statements},
    },
    func: {
        func_0: {...statements},
        func_1: {...statements},
    },
    .
    .
    .
}
"""

# TODO: CFG Printer
class CFGPrinter:
    ...