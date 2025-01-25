from itertools import chain
from typing import Any, Dict, List, Optional, Generator, TypeVar, Set, Type, get_origin, get_args, Union, get_type_hints
from collections.abc import Sequence
from inspect import signature

from .schemas import Var, TypedVar, State, Goal, FunctionType

T = TypeVar('T')

def walk(var: Any, subs: Dict[Var, Any]) -> Any:
    """Walk through the substitution chain to find the final value.
    
    Args:
        var: The variable or value to resolve
        subs: Dictionary of variable substitutions
    
    Returns:
        The final value after following all substitutions
    """
    seen = set()  # Prevent infinite recursion
    while isinstance(var, Var) and var in subs and var not in seen:
        seen.add(var)
        var = subs[var]
    return var

def collect_vars(val: Any) -> Set[Var]:
    """Collect all variables from a value.
    
    Args:
        val: The value to collect variables from
    
    Returns:
        Set of variables found in the value
    """
    vars_set = set()
    if isinstance(val, Var):
        vars_set.add(val)
    elif isinstance(val, (list, tuple)):
        for v in val:
            vars_set.update(collect_vars(v))
    elif isinstance(val, dict):
        for k, v in val.items():
            vars_set.update(collect_vars(k))
            vars_set.update(collect_vars(v))
    return vars_set

def deep_walk(val: Any, subs: Dict[Var, Any], depth: int = 0) -> Any:
    """Recursively walk through nested structures.
    
    Args:
        val: The value to resolve, which may contain nested variables
        subs: Dictionary of variable substitutions
        depth: Current recursion depth (used to prevent infinite recursion)
    
    Returns:
        The resolved value with all variables substituted
    """
    if depth > 100:  # Prevent infinite recursion
        return val
    
    val = walk(val, subs)
    
    if isinstance(val, (list, tuple)):
        return type(val)(deep_walk(v, subs, depth + 1) for v in val)
    
    return val

def unify(u: Any, v: Any, subs: Dict[Var, Any]) -> Optional[Dict[Var, Any]]:
    """Unify two terms with improved type handling."""
    u = walk(u, subs)
    v = walk(v, subs)
    
    if u == v:
        return subs
    
    # Handle TypedVar constraints
    if isinstance(u, TypedVar):
        if isinstance(v, Var):
            # Create a new TypedVar with the same type constraints
            typed_var = TypedVar(v.name, u.type, u.nullable)
            return {**subs, v: typed_var}
        if isinstance(v, TypedVar):
            if not u.check_type(v):
                return None
            return subs
        if isinstance(v, (list, tuple)):
            # For lists/tuples, handle element type constraints
            origin_type = get_origin(u.type[0])
            if origin_type in (list, tuple):
                elem_type = get_args(u.type[0])[0]
                new_subs = subs.copy()
                # Apply type constraints to all variables in the list
                for elem in v:
                    elem_val = walk(elem, new_subs)
                    if isinstance(elem_val, Var):
                        # Create a new TypedVar for the element
                        elem_typed_var = TypedVar(elem_val.name, elem_type)
                        if (new_subs := unify(elem_typed_var, elem_val, new_subs)) is None:
                            return None
                return {**new_subs, u: v}
            # Check type after propagating constraints to variables
            if not u.check_type(v):
                return None
            return {**subs, u: v}
        if not u.check_type(v):
            return None
        return {**subs, u: v}
    
    if isinstance(v, TypedVar):
        return unify(v, u, subs)
    
    # Handle nested structures
    if isinstance(u, (list, tuple)) and isinstance(v, (list, tuple)) and len(u) == len(v):
        new_subs = subs.copy()
        for ui, vi in zip(u, v):
            if (new_subs := unify(ui, vi, new_subs)) is None:
                return None
        return new_subs
    
    # Regular Var handling
    if isinstance(u, Var):
        # Check if we're unifying with a value that should satisfy a type constraint
        u_val = walk(u, subs)
        if isinstance(u_val, TypedVar) and not u_val.check_type(v):
            return None
        return {**subs, u: v}
    if isinstance(v, Var):
        # Check if we're unifying with a value that should satisfy a type constraint
        v_val = walk(v, subs)
        if isinstance(v_val, TypedVar) and not v_val.check_type(u):
            return None
        return {**subs, v: u}
    
    return None

def eq(a: Any, b: Any) -> Goal:
    """Create a goal that unifies two terms.
    
    Args:
        a: First term to unify
        b: Second term to unify
    
    Returns:
        A goal function that attempts to unify the terms
    """
    def goal(state: State) -> Generator[State, None, None]:
        if new_subs := unify(a, b, state.subs):
            new_state = state.copy()
            new_state.subs = new_subs
            if all(c(new_state) for c in new_state.constraints):
                yield new_state
    return goal

def run(goals: List[Goal], n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run a list of goals and return solutions.
    
    Args:
        goals: List of goals to satisfy
        n: Maximum number of solutions to return (None for all solutions)
    
    Returns:
        List of solutions, where each solution is a dictionary mapping
        variable names to their values
    """
    Var._id = 0
    
    # Collect all variables from the goals
    original_vars = set()
    for goal in goals:
        if hasattr(goal, '__closure__') and goal.__closure__:
            for cell in goal.__closure__:
                if isinstance(cell.cell_contents, tuple):
                    for item in cell.cell_contents:
                        original_vars.update(collect_vars(item))
                else:
                    original_vars.update(collect_vars(cell.cell_contents))
    
    states = [State({}, [])]
    
    # Process goals in sequence
    for goal in goals:
        states = list(chain.from_iterable(goal(s) for s in states))
        if not states:
            break
    
    solutions = []
    for state in states[:n]:
        sol = {}
        
        # Resolve each original variable
        for var in original_vars:
            val = deep_walk(var, state.subs)
            if isinstance(val, TypedVar):
                # If it's a TypedVar with a concrete value, use that
                concrete_val = walk(val, state.subs)
                if not isinstance(concrete_val, (TypedVar, Var)):
                    sol[var.name] = concrete_val
            elif not isinstance(val, Var):
                sol[var.name] = val
            
            # Also include type information for variables that got type constraints
            type_val = walk(var, state.subs)
            if isinstance(type_val, TypedVar):
                type_str = str(type_val.type[0])
                if get_origin(type_val.type[0]) is not None:
                    type_str = str(type_val.type[0].__origin__.__name__)
                    args = get_args(type_val.type[0])
                    if args:
                        type_str += f"[{', '.join(arg.__name__ for arg in args)}]"
                else:
                    type_str = type_val.type[0].__name__
                sol[f"{var.name}_type"] = type_str
        
        if sol:
            solutions.append(sol)
    
    return solutions

def getitem(container: Any, key: Any, result: Var) -> Goal:
    """Create a goal that unifies result with container[key].
    
    Args:
        container: The container to index into
        key: The key/index to use
        result: Variable to unify with the result
        
    Returns:
        A goal that attempts to get the item and unify it
    """
    def goal(state: State) -> Generator[State, None, None]:
        container_val = walk(container, state.subs)
        key_val = walk(key, state.subs)
        
        if isinstance(container_val, (list, tuple, dict)) and not isinstance(key_val, Var):
            try:
                item = container_val[key_val]
                if new_subs := unify(result, item, state.subs):
                    new_state = state.copy()
                    new_state.subs = new_subs
                    if all(c(new_state) for c in new_state.constraints):
                        yield new_state
            except (IndexError, KeyError):
                pass
    return goal

def infer_type(value: Any) -> Union[Type, str, FunctionType]:
    """Infer the type of a value with support for generic types and functions."""
    # Handle functions
    if callable(value):
        # Try to get type hints if available
        hints = get_type_hints(value)
        if hints:
            inputs = []
            for param in signature(value).parameters.values():
                if param.name in hints:
                    inputs.append(hints[param.name])
                else:
                    inputs.append(Any)
            output = hints.get('return', Any)
            return str(FunctionType(inputs, output))  # Convert to string immediately
        return "Callable"
        
    # Handle TypedVar values by using their type constraint
    if isinstance(value, TypedVar):
        type_str = str(value.type[0])
        if get_origin(value.type[0]) is not None:
            type_str = str(value.type[0].__origin__.__name__)
            args = get_args(value.type[0])
            if args:
                type_str += f"[{', '.join(arg.__name__ for arg in args)}]"
        else:
            type_str = value.type[0].__name__
        return type_str

    if isinstance(value, dict):
        if not value:
            return "Dict"
        # Infer key and value types
        key_types = {infer_type(k) for k in value.keys()}
        value_types = {infer_type(v) for v in value.values()}
        
        # Convert type objects to strings if needed
        key_types = {t.__name__ if isinstance(t, type) else t for t in key_types}
        value_types = {t.__name__ if isinstance(t, type) else t for t in value_types}
        
        key_type = next(iter(key_types)) if len(key_types) == 1 else f"Union[{', '.join(sorted(key_types))}]"
        value_type = next(iter(value_types)) if len(value_types) == 1 else f"Union[{', '.join(sorted(value_types))}]"
        
        return f"Dict[{key_type}, {value_type}]"
        
    if isinstance(value, set):
        if not value:
            return "Set"
        element_types = {infer_type(x) for x in value}
        element_types = {t.__name__ if isinstance(t, type) else t for t in element_types}
        
        if len(element_types) == 1:
            return f"Set[{next(iter(element_types))}]"
        return f"Set[Union[{', '.join(sorted(element_types))}]]"
        
    if isinstance(value, (list, tuple)):
        container_type = type(value)
        if not value:
            return container_type.__name__
            
        # Recursively infer types for elements
        element_types = set()
        for x in value:
            inferred = infer_type(x)
            # Convert FunctionType to string if needed
            if isinstance(inferred, FunctionType):
                element_types.add(str(inferred))
            else:
                element_types.add(inferred)
        
        # Convert type objects to strings if needed
        element_types = {t.__name__ if isinstance(t, type) else t for t in element_types}
        
        if len(element_types) == 1:
            elem_type = next(iter(element_types))
            if container_type is list:
                return f"List[{elem_type}]"
            return f"Tuple[{elem_type}, ...]"
        
        types_str = ", ".join(sorted(element_types))
        if container_type is list:
            return f"List[Union[{types_str}]]"
        return f"Tuple[{types_str}]"
        
    return type(value).__name__

def type_of(var: Var, type_var: Optional[Var] = None) -> Goal:
    """Create a goal that unifies a variable with its inferred type."""
    def goal(state: State) -> Generator[State, None, None]:
        val = walk(var, state.subs)
        if isinstance(val, Var):
            yield state  # Can't infer type yet
        else:
            inferred = infer_type(val)
            new_state = state.copy()
            
            if type_var is not None:
                if new_subs := unify(type_var, inferred, new_state.subs):
                    new_state.subs = new_subs
                    yield new_state
            else:
                new_state.subs[var] = inferred
                yield new_state
                
    return goal

def function_type(func: Var, inputs: List[Type], output: Type) -> Goal:
    """Create a goal that constrains a variable to be a function with given types."""
    def goal(state: State) -> Generator[State, None, None]:
        func_val = walk(func, state.subs)
        if isinstance(func_val, Var):
            # Create a TypedVar with function type
            typed_var = TypedVar(func_val.name, FunctionType(inputs, output))
            if new_subs := unify(func_val, typed_var, state.subs):
                new_state = state.copy()
                new_state.subs = new_subs
                yield new_state
        elif callable(func_val):
            # Check if function matches type constraints
            hints = get_type_hints(func_val)
            if hints:
                actual_inputs = []
                for param in signature(func_val).parameters.values():
                    if param.name in hints:
                        actual_inputs.append(hints[param.name])
                actual_output = hints.get('return', Any)
                if all(issubclass(a, e) for a, e in zip(actual_inputs, inputs)) and \
                   issubclass(actual_output, output):
                    yield state
    return goal