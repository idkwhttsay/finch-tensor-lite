from ..finch_assembly.nodes import *
from typing import List, Optional

# TODO: Do a system design for how to build Basic Blocks and CFG based on the AST. Understand how to handle loops, breaks, continues, function calls, etc. \
# and how to represent them in the CFG. Also, figure out a better way of representiang the connections between blocks (probably a map will work best).

class BasicBlock:
    """
    Represents a basic block in a CFG.
    """
    
    _next_id = 0
    
    def __init__(self):
        self.id = BasicBlock._next_id
        BasicBlock._next_id += 1
        self.statements: List[AssemblyNode] = []
        self.predecessors: List[BasicBlock] = []
        self.successors: List[BasicBlock] = []
        self.is_terminated = False  # Whether block ends with terminator (return, break, etc.)
    
    def add_statement(self, stmt: AssemblyNode):
        self.statements.append(stmt)

    def add_successor(self, successor: 'BasicBlock'):
        if successor not in self.successors:
            self.successors.append(successor)
        
        if self not in successor.predecessors:
            successor.predecessors.append(self)

class ControlFlowGraph:
    """
    Represents a CFG containing basic blocks.
    """
    
    def __init__(self, entry_block: BasicBlock):
        self.entry_block = entry_block
        self.blocks: List[BasicBlock] = []
    
    def add_block(self, block: BasicBlock):
        """Add a basic block to the CFG."""
        if block not in self.blocks:
            self.blocks.append(block)

class BasicBlockBuilder:
    """
    Builds a control flow graph from an assembly AST by creating basic blocks.
    """
    
    def __init__(self):
        self.blocks: List[BasicBlock] = []
        self.current_block: Optional[BasicBlock] = None
        self.loop_header_stack: List[BasicBlock] = []  # For handling nested loops
        self.loop_exit_stack: List[BasicBlock] = []    # For handling breaks
    
    def create_block(self) -> BasicBlock:
        """Create a new basic block and add it to the CFG."""
        block = BasicBlock()
        self.blocks.append(block)
        return block

    def start_new_block(self) -> BasicBlock:
        """Start a new basic block, making it the current block."""
        block = self.create_block()
        self.current_block = block
        return block
    
    def build_cfg(self, root: AssemblyNode) -> ControlFlowGraph:
        """Build a control flow graph from the root assembly node."""
        self.blocks = []
        self.current_block = None
        self.loop_header_stack = []
        self.loop_exit_stack = []
        
        # Start with entry block
        entry_block = self.start_new_block()
        
        # Process the root node
        self._process_node(root)
        
        # Create CFG
        cfg = ControlFlowGraph(entry_block)
        cfg.blocks = self.blocks
        cfg.find_exit_blocks()
        
        return cfg
    
    def _process_node(self, node: AssemblyNode) -> Optional[BasicBlock]:
        """
        Process an assembly node and add it to the appropriate basic block.
        Returns the block that execution continues from (None if execution terminates).
        """
        if self.current_block is None:
            raise RuntimeError("No current basic block to add statements to")
        
        match node:
            # Simple statements - add to current block
            case Assign(lhs, rhs):
                self.current_block.add_statement(node)
                return self.current_block
            case Store(buf, idx, val):
                self.current_block.add_statement(node)
                return self.current_block
            case Resize(buf, size):
                self.current_block.add_statement(node)
                return self.current_block
            case Unpack(slot, val):
                self.current_block.add_statement(node)
                return self.current_block
            case Repack(slot):
                self.current_block.add_statement(node)
                return self.current_block
            case SetAttr(obj, attr, val):
                self.current_block.add_statement(node)
                return self.current_block
            
            # Expressions that can be standalone statements
            case Call(op, args):
                self.current_block.add_statement(node)
                return self.current_block
            
            # Terminators - end current block
            case Return(value):
                self.current_block.add_statement(node)
                self.current_block.is_terminated = True
                return None  # Execution terminates
            
            case Break():
                self.current_block.add_statement(node)
                # Connect to loop exit if we're in a loop
                if self.loop_exit_stack:
                    loop_exit = self.loop_exit_stack[-1]
                    self.current_block.add_successor(loop_exit)
                self.current_block.is_terminated = True
                return None  # Execution terminates
            
            # Control flow constructs
            case Block(bodies):
                return self._process_block(bodies)
            case If(condition, body):
                return self._process_if(condition, body)
            case IfElse(condition, then_body, else_body):
                return self._process_if_else(condition, then_body, else_body)
            case WhileLoop(condition, body):
                return self._process_while_loop(condition, body)
            case ForLoop(var, start, end, body):
                return self._process_for_loop(var, start, end, body)
            case BufferLoop(buffer, var, body):
                return self._process_buffer_loop(buffer, var, body)
            case Function(name, args, body):
                return self._process_function(name, args, body)
            case Module(funcs):
                return self._process_module(funcs)
            
            # Expressions and other nodes that don't create statements by themselves
            case Literal(_) | Variable(_, _) | GetAttr(_, _) | Load(_, _) | Length(_) | Slot(_, _) | Stack(_, _):
                return self.current_block
            case _:
                raise NotImplementedError(f"CFG construction not implemented for {type(node).__name__}")
    
    def _process_block(self, bodies: tuple) -> Optional[BasicBlock]:
        """Process a sequence of statements in a block."""
        current = self.current_block
        
        for body in bodies:
            if current is None:
                # Previous statement terminated execution
                break
            self.current_block = current
            current = self._process_node(body)
        
        return current
    
    def _process_if(self, condition: AssemblyExpression, body: AssemblyNode) -> BasicBlock:
        """Process an if statement."""
        # Current block ends with the condition evaluation
        if_block = self.current_block
        if_block.add_statement(If(condition, Block(())))  # Placeholder for the condition
        
        # Create then block
        then_block = self.start_new_block()
        if_block.add_successor(then_block)
        
        # Process then body
        after_then = self._process_node(body)
        
        # Create merge block
        merge_block = self.create_block()

        # Connect then block to merge (if it doesn't terminate)
        if after_then is not None:
            after_then.add_successor(merge_block)
        
        # Connect if block directly to merge (else path)
        if_block.add_successor(merge_block)
        
        self.current_block = merge_block
        return merge_block
    
    def _process_if_else(self, condition: AssemblyExpression, then_body: AssemblyNode, else_body: AssemblyNode) -> BasicBlock:
        """Process an if-else statement."""
        # Current block ends with the condition evaluation
        if_block = self.current_block
        if_block.add_statement(IfElse(condition, Block(()), Block(())))  # Placeholder
        
        # Create then block
        then_block = self.start_new_block("if_then")
        if_block.add_successor(then_block)
        after_then = self._process_node(then_body)
        
        # Create else block
        else_block = self.start_new_block("if_else")
        if_block.add_successor(else_block)
        after_else = self._process_node(else_body)
        
        # Create merge block
        merge_block = self.create_block("if_merge")
        
        # Connect both branches to merge (if they don't terminate)
        if after_then is not None:
            after_then.add_successor(merge_block)
        if after_else is not None:
            after_else.add_successor(merge_block)
        
        self.current_block = merge_block
        return merge_block
    
    def _process_while_loop(self, condition: AssemblyExpression, body: AssemblyNode) -> BasicBlock:
        """Process a while loop."""
        # Create loop header block for condition evaluation
        loop_header = self.create_block()
        self.current_block.add_successor(loop_header)
        
        # Create loop body block
        loop_body = self.create_block()

        # Create loop exit block
        loop_exit = self.create_block()

        # Connect header to body and exit
        loop_header.add_successor(loop_body)
        loop_header.add_successor(loop_exit)
        loop_header.add_statement(WhileLoop(condition, Block(())))  # Placeholder
        
        # Process body with loop context
        self.loop_header_stack.append(loop_header)
        self.loop_exit_stack.append(loop_exit)
        
        self.current_block = loop_body
        after_body = self._process_node(body)
        
        # Connect body back to header (if it doesn't terminate)
        if after_body is not None:
            after_body.add_successor(loop_header)
        
        self.loop_header_stack.pop()
        self.loop_exit_stack.pop()
        
        self.current_block = loop_exit
        return loop_exit
    
    def _process_for_loop(self, var: Variable, start: AssemblyExpression, end: AssemblyExpression, body: AssemblyNode) -> BasicBlock:
        """Process a for loop."""
        # Similar to while loop but with initialization
        # Add initialization to current block
        self.current_block.add_statement(Assign(var, start))
        
        # Create loop header for condition check
        loop_header = self.create_block("for_header")
        self.current_block.add_successor(loop_header)
        
        # Create loop body and exit blocks
        loop_body = self.create_block("for_body")
        loop_exit = self.create_block("for_exit")
        
        # Connect header to body and exit
        loop_header.add_successor(loop_body)
        loop_header.add_successor(loop_exit)
        
        # Process body with loop context
        self.loop_header_stack.append(loop_header)
        self.loop_exit_stack.append(loop_exit)
        
        self.current_block = loop_body
        after_body = self._process_node(body)
        
        # Add increment and connect back to header
        if after_body is not None:
            # Add increment (simplified)
            after_body.add_statement(ForLoop(var, start, end, Block(())))  # Placeholder
            after_body.add_successor(loop_header)
        
        self.loop_header_stack.pop()
        self.loop_exit_stack.pop()
        
        self.current_block = loop_exit
        return loop_exit
    
    def _process_buffer_loop(self, buffer: AssemblyExpression, var: Variable, body: AssemblyNode) -> BasicBlock:
        """Process a buffer loop (similar to for loop)."""
        # Similar structure to for loop
        loop_header = self.create_block("buffer_loop_header")
        self.current_block.add_successor(loop_header)
        
        loop_body = self.create_block("buffer_loop_body")
        loop_exit = self.create_block("buffer_loop_exit")
        
        loop_header.add_successor(loop_body)
        loop_header.add_successor(loop_exit)
        loop_header.add_statement(BufferLoop(buffer, var, Block(())))  # Placeholder
        
        # Process body with loop context
        self.loop_header_stack.append(loop_header)
        self.loop_exit_stack.append(loop_exit)
        
        self.current_block = loop_body
        after_body = self._process_node(body)
        
        if after_body is not None:
            after_body.add_successor(loop_header)
        
        self.loop_header_stack.pop()
        self.loop_exit_stack.pop()
        
        self.current_block = loop_exit
        return loop_exit
    
    def _process_function(self, name: Variable, args: tuple, body: AssemblyNode) -> BasicBlock:
        """Process a function definition."""
        # Functions create their own separate CFG
        # For now, just process the body in the current context
        self.current_block.add_statement(Function(name, args, Block(())))  # Placeholder
        return self._process_node(body)
    
    def _process_module(self, funcs: tuple) -> BasicBlock:
        """Process a module with multiple functions."""
        current = self.current_block
        for func in funcs:
            if current is None:
                break
            self.current_block = current
            current = self._process_node(func)
        return current

def create_test_ast():
    # Variables
    x = Variable("x", int)
    y = Variable("y", int)
    condition = Variable("condition", bool)

    # Simple assignments
    assign1 = Assign(x, Literal(5))
    assign2 = Assign(y, Literal(10))
    assign3 = Assign(x, Literal(15))

    # Control flow statements
    return_stmt = Return(x)
    if_else_stmt = IfElse(condition, assign2, assign3)
    for_loop = ForLoop(Variable("i", int), Literal(0), Literal(10), assign3)

    # Create a block with mixed statements
    test_block = Block((assign1, if_else_stmt, for_loop, return_stmt))

    return test_block