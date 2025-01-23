from typing import Any, Dict, List, Callable, Generator
from dataclasses import dataclass

class Var:
    _id = 0
    def __init__(self):
        self.id = Var._id
        Var._id += 1
    
    def __repr__(self):
        return f"_{self.id}"

class State:
    def __init__(self, subs: Dict[Var, Any], constraints: List[Callable[['State'], bool]]):
        self.subs = subs
        self.constraints = constraints
    
    def copy(self):
        return State(self.subs.copy(), self.constraints.copy())

# Type definitions
Goal = Callable[[State], Generator[State, None, None]] 