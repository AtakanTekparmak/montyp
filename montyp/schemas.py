from typing import Any, Dict, List, Callable, Generator, TypeVar
from dataclasses import dataclass
import inspect
import sys

T = TypeVar('T')  # Generic type for State constraints

class Var:
    """A logical variable that can be unified with any value.
    
    Each variable has a unique ID that is used for comparison and a name for representation.
    """
    _id: int = 0
    
    def __init__(self, name: str) -> None:
        """Create a new variable with a unique ID and given name.
        
        Args:
            name: The name of the variable for representation
        """
        self.id: int = Var._id
        Var._id += 1
        self.name = name
    
    def __repr__(self) -> str:
        """Return a string representation of the variable."""
        return self.name

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