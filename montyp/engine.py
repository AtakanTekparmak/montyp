from itertools import chain
from typing import Any, Dict, List, Optional, Generator, TypeVar, Set, Type, get_origin, get_args, Union, get_type_hints, Callable
from collections.abc import Sequence
from inspect import signature
from functools import partial
import inspect

from .schemas import Var, TypedVar, State, Goal, FunctionType, AbstractFunctionType, LogicalFunction

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
        
    # Handle Var unification
    if isinstance(u, Var) and not isinstance(u, TypedVar):
        return {**subs, u: v}
    if isinstance(v, Var) and not isinstance(v, TypedVar):
        return {**subs, v: u}
        
    # Handle TypedVar constraints
    if isinstance(u, TypedVar):
        if not u.check_type(v):
            return None
        return {**subs, u: v}
    if isinstance(v, TypedVar):
        if not v.check_type(u):
            return None
        return {**subs, v: u}
    
    # Handle LogicalFunction
    if isinstance(u, LogicalFunction) and callable(v):
        try:
            hints = get_type_hints(v)
            if hints:
                param_type = next(iter(hints.values()))
                return_type = hints.get('return')
                if param_type != u.input_type or return_type != u.output_type:
                    return None
                    
            for input_val, output_val in u._examples:
                if v(input_val) != output_val:
                    return None
            return subs
        except:
            return None
    
    if isinstance(v, LogicalFunction) and callable(u):
        return unify(u, v, subs)

    # Handle collections
    if isinstance(u, (list, tuple)) and isinstance(v, (list, tuple)) and len(u) == len(v):
        new_subs = subs
        for ui, vi in zip(u, v):
            if (new_subs := unify(ui, vi, new_subs)) is None:
                return None
        return new_subs
        
    # Handle function type constraints
    if isinstance(u, FunctionType) and callable(v):
        try:
            hints = get_type_hints(v)
            if not hints:
                return None
                
            # Get parameter types from function signature
            sig = signature(v)
            param_types = [hints[p.name] for p in sig.parameters.values()]
            return_type = hints.get('return')
            
            # Check number of parameters matches
            if len(param_types) != len(u.inputs):
                return None
                
            # Check each parameter type matches exactly
            for param_type, expected_type in zip(param_types, u.inputs):
                if param_type != expected_type:
                    return None
                
            # Check return type matches exactly
            if return_type != u.output:
                return None
                
            return subs
        except Exception as e:
            return None
            
    if isinstance(v, FunctionType) and callable(u):
        return unify(v, u, subs)
        
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
        raw_sol = {}  # Store raw values before JSON conversion
        
        # Resolve each original variable
        for var in original_vars:
            val = deep_walk(var, state.subs)
            raw_sol[var.name] = val  # Store raw value
            
            if isinstance(val, TypedVar):
                # If it's a TypedVar with a concrete value, use that
                concrete_val = walk(val, state.subs)
                if not isinstance(concrete_val, (TypedVar, Var)):
                    sol[var.name] = concrete_val
                # Always include type information for TypedVar
                type_str = str(val.type[0])
                if get_origin(val.type[0]) is not None:
                    type_str = str(val.type[0].__origin__.__name__)
                    args = get_args(val.type[0])
                    if args:
                        type_str += f"[{', '.join(arg.__name__ for arg in args)}]"
                else:
                    type_str = val.type[0].__name__
                sol[f"{var.name}_type"] = type_str
            elif isinstance(val, LogicalFunction):
                # Only include type information for LogicalFunction
                sol[f"{var.name}_type"] = f"({val.input_type.__name__}) -> {val.output_type.__name__}"
            elif not isinstance(val, Var):
                # Convert FunctionType to string representation
                if isinstance(val, FunctionType):
                    sol[var.name] = str(val)
                else:
                    sol[var.name] = val
                
                # Include inferred type information
                inferred = infer_type(val)
                if isinstance(inferred, (str, FunctionType)):
                    sol[f"{var.name}_type"] = str(inferred)
                elif hasattr(inferred, '__name__'):
                    sol[f"{var.name}_type"] = inferred.__name__
        
        if sol:
            solutions.append({**sol, "_raw": raw_sol})  # Include raw values for testing
    
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

def get_higher_order_type(func: Callable) -> Optional[str]:
    """
    Get the type signature for higher-order functions.
    
    Args:
        func: The higher-order function to get the type signature for
    
    Returns:
        The type signature as a string, or None if the function is not recognized
    """
    if func == map:
        return "HigherOrder[map]"
    # Add other higher-order functions as needed
    return None

def apply_higher_order(func: Callable, args: List[Any], result: Any) -> Any:
    """
    Apply higher-order function and handle type inference.
    
    Args:
        func: The higher-order function to apply
        args: The arguments to the function
        result: The expected result of the function application
    
    Returns:
        The result of the function application, or None if the function is not recognized
    """
    if func == map:
        f, iterable = args
        
        # Handle case where function is a variable
        if isinstance(f, (Var, TypedVar)):
            return None, None  # Let the regular unification handle this case
            
        # Get type hints for the mapping function
        try:
            f_hints = get_type_hints(f)
            input_type = f_hints.get('x') or next(iter(f_hints.values()))  # Get first param type
            return_type = f_hints.get('return')
        except (TypeError, ValueError):
            return None, None
        
        # Apply map and convert to list
        try:
            mapped = list(map(f, iterable))
            
            # Infer result type based on function signature
            if return_type:
                return mapped, f'List[{return_type.__name__}]'
            return mapped, f'List[{type(mapped[0]).__name__}]' if mapped else 'List'
        except (TypeError, ValueError):
            return None, None
    
    return None, None

def apply(func: Callable, args: List[Any], result: Var) -> Goal:
    """
    Create a goal that applies a function to arguments and unifies the result.
    
    Args:
        func: The function to apply
        args: The arguments to the function
        result: The variable to unify with the result of the function application
    
    Returns:
        A goal that attempts to apply the function and unify the result
    """
    def goal(state: State) -> Generator[State, None, None]:
        walked_args = [walk(arg, state.subs) for arg in args]
        walked_result = walk(result, state.subs)

        # Special handling for map function
        if func == map:
            map_func, iterable = walked_args
            new_state = state.copy()
            walked_iterable = walk(iterable, state.subs)

            # Handle case where iterable is TypedVar
            if isinstance(walked_iterable, TypedVar):
                walked_iterable = []  # Empty list matching the type constraint

            # If we have concrete input and output, we can infer the function
            if not isinstance(walked_result, (Var, TypedVar)):
                # First try to find a concrete function that matches
                found = False
                
                # Try the function if it's already bound
                if not isinstance(map_func, (Var, TypedVar)):
                    try:
                        test_result = list(map(map_func, walked_iterable))
                        if test_result == walked_result:
                            if new_subs := unify(result, test_result, new_state.subs):
                                new_state.subs.update(new_subs)
                                if all(c(new_state) for c in new_state.constraints):
                                    yield new_state
                                    found = True
                    except:
                        pass
                
                # If not found and map_func is a variable, try to deduce it
                if not found and isinstance(map_func, Var):
                    # Create a logical function from the examples
                    input_type = (type(walked_iterable[0]) if walked_iterable 
                                else get_args(iterable.type[0])[0] if isinstance(iterable, TypedVar) 
                                else int)  # fallback
                    output_type = type(walked_result[0]) if walked_result else str  # fallback
                    logical_func = LogicalFunction(map_func.name, input_type, output_type)
                    
                    if walked_iterable and walked_result:
                        for x, y in zip(walked_iterable, walked_result):
                            logical_func.add_example(x, y)
                            
                    candidate_state = new_state.copy()
                    if func_subs := unify(map_func, logical_func, candidate_state.subs):
                        candidate_state.subs.update(func_subs)
                        if result_subs := unify(result, walked_result, candidate_state.subs):
                            candidate_state.subs.update(result_subs)
                            if all(c(candidate_state) for c in candidate_state.constraints):
                                yield candidate_state
            else:
                # Handle case where result is a variable
                if not isinstance(map_func, (Var, TypedVar)):
                    try:
                        mapped_result = list(map(map_func, walked_iterable))
                        if new_subs := unify(result, mapped_result, new_state.subs):
                            new_state.subs.update(new_subs)
                            if all(c(new_state) for c in new_state.constraints):
                                yield new_state
                    except:
                        pass
            return

        # Handle regular function application
        if not isinstance(func, (Var, TypedVar)):
            try:
                # Handle case where some arguments are variables
                if any(isinstance(arg, Var) for arg in walked_args):
                    # Try to deduce variable arguments from result
                    if not isinstance(walked_result, (Var, TypedVar)):
                        # Create a logical function to solve for the variable
                        var_idx = next(i for i, arg in enumerate(walked_args) if isinstance(arg, Var))
                        var = walked_args[var_idx]
                        
                        # Try possible values that would give the expected result
                        for test_val in range(-1000, 1001):  # Reasonable range for integers
                            test_args = list(walked_args)
                            test_args[var_idx] = test_val
                            try:
                                if func(*test_args) == walked_result:
                                    candidate_state = state.copy()
                                    if var_subs := unify(var, test_val, candidate_state.subs):
                                        candidate_state.subs.update(var_subs)
                                        if all(c(candidate_state) for c in candidate_state.constraints):
                                            yield candidate_state
                                            break
                            except:
                                continue
                else:
                    # Regular function application with concrete arguments
                    applied_result = func(*walked_args)
                    if new_subs := unify(result, applied_result, state.subs):
                        new_state = state.copy()
                        new_state.subs = new_subs
                        if all(c(new_state) for c in new_state.constraints):
                            yield new_state
            except:
                pass
    return goal

def infer_type(value: Any) -> Union[Type, str, FunctionType]:
    """Infer the type of a value."""
    # Handle LogicalFunction
    if isinstance(value, LogicalFunction):
        return FunctionType([value.input_type], value.output_type)
        
    # Handle higher-order functions
    if value == map:
        return "HigherOrder[map]"
        
    # Handle functions
    if callable(value):
        try:
            hints = get_type_hints(value)
            if hints:
                inputs = []
                for param in signature(value).parameters.values():
                    if param.name in hints:
                        inputs.append(hints[param.name])
                    else:
                        inputs.append(Any)
                output = hints.get('return', Any)
                return FunctionType(inputs, output)
        except TypeError:
            pass
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

def type_of(var: Any, type_var: Optional[Var] = None) -> Goal:
    """
    Create a goal that unifies a variable with its inferred type.
    
    Args:
        var: The variable to unify with its inferred type
        type_var: An optional variable to unify with the inferred type
    
    Returns:
        A goal that attempts to unify the variable with its inferred type
    """
    def goal(state: State) -> Generator[State, None, None]:
        val = walk(var, state.subs)
        
        # Special handling for TypedVar
        if isinstance(val, TypedVar):
            type_str = str(val.type[0])
            if get_origin(val.type[0]) is not None:
                type_str = str(val.type[0].__origin__.__name__)
                args = get_args(val.type[0])
                if args:
                    type_str += f"[{', '.join(arg.__name__ for arg in args)}]"
            else:
                if hasattr(val.type[0], '__name__'):
                    type_str = val.type[0].__name__
                elif hasattr(val.type[0], '__repr__'):
                    type_str = val.type[0].__repr__()
                else:
                    type_str = str(val.type[0])
                
            new_state = state.copy()
            if type_var is not None:
                if new_subs := unify(type_var, type_str, new_state.subs):
                    new_state.subs = new_subs
                    yield new_state
            else:
                new_state.subs[var] = type_str
                yield new_state
            return
            
        if isinstance(val, Var):
            yield state  # Can't infer type yet
            return
            
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
    """
    Create a goal that constrains a variable to be a function with given types.
    
    Args:
        func: The variable to constrain to be a function
        inputs: The expected input types for the function
        output: The expected output type for the function
    
    Returns:
        A goal that attempts to constrain the variable to be a function with the given types
    """
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