from typing import Any, Dict, List, Callable, Generator, TypeVar
from dataclasses import dataclass

T = TypeVar('T')  # Generic type for State constraints

class Var:
    """A logical variable that can be unified with any value.
    
    Each variable has a unique ID that is used for representation and comparison.
    Variables are automatically numbered starting from 0.
    """
    _id: int = 0
    
    def __init__(self) -> None:
        """Create a new variable with a unique ID."""
        self.id: int = Var._id
        Var._id += 1
    
    def __repr__(self) -> str:
        """Return a string representation of the variable (e.g., '_0', '_1')."""
        return f"_{self.id}"

class State:
    """Represents the current state of the logical computation.
    
    Contains:
        - substitutions: mappings from variables to their values
        - constraints: predicates that must hold true for the state to be valid
    """
    
    def __init__(self, subs: Dict[Var, Any], constraints: List[Callable[['State'], bool]]) -> None:
        """Initialize a new state.
        
        Args:
            subs: Dictionary mapping variables to their values
            constraints: List of functions that take a state and return whether it's valid
        """
        self.subs = subs
        self.constraints = constraints
    
    def copy(self) -> 'State':
        """Create a deep copy of the state.
        
        Returns:
            A new State with copied substitutions and constraints.
        """
        return State(self.subs.copy(), self.constraints.copy())

# Type definitions
Goal = Callable[[State], Generator[State, None, None]]