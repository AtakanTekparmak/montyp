from typing import Any, Dict, List, Callable, Generator, TypeVar, Type, Optional, Union, get_origin, get_args
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

class TypedVar(Var):
    """A logical variable with type constraints."""
    
    def __init__(self, name: str, type_: Union[Type, tuple[Type, ...]], nullable: bool = False) -> None:
        """Create a new typed variable.
        
        Args:
            name: The name of the variable
            type_: The expected type(s) for this variable
            nullable: Whether None is an acceptable value
        """
        super().__init__(name)
        self.type = type_ if isinstance(type_, tuple) else (type_,)
        self.nullable = nullable
    
    def check_type(self, value: Any) -> bool:
        """Check if a value matches the variable's type constraints."""
        if value is None:
            return self.nullable
            
        def check_container_type(container_value: Any, container_type: Type, elem_type: Type) -> bool:
            origin_type = get_origin(container_type) or container_type
            if not isinstance(container_value, origin_type):
                return False
                
            if not container_value:  # Empty container
                return True
                
            # Handle Union types in element type
            if get_origin(elem_type) is Union:
                elem_types = get_args(elem_type)
                return all(any(isinstance(elem, t) for t in elem_types)
                          for elem in container_value)
                          
            # Handle nested generic types
            if get_origin(elem_type) is not None:
                return all(check_container_type(elem, get_origin(elem_type), get_args(elem_type)[0])
                          for elem in container_value)
            
            # Handle regular types
            return all(isinstance(elem, elem_type) for elem in container_value)
            
        # Handle generic container types
        for t in self.type:
            origin = get_origin(t)
            if origin is not None:
                # It's a generic type (like List[int])
                type_args = get_args(t)
                if not type_args:
                    return isinstance(value, origin)
                    
                if isinstance(value, (list, tuple)):
                    return check_container_type(value, origin, type_args[0])
                    
                return isinstance(value, origin)
                
            # Regular type check
            if isinstance(value, t):
                return True
                
        return False
    
    def __repr__(self) -> str:
        def type_name(t):
            origin = get_origin(t)
            if origin is not None:
                args = get_args(t)
                if args:
                    args_str = ', '.join(arg.__name__ for arg in args)
                    return f"{origin.__name__}[{args_str}]"
            return t.__name__
            
        type_str = ' | '.join(type_name(t) for t in self.type)
        if self.nullable:
            type_str += ' | None'
        return f"{self.name}: {type_str}"

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