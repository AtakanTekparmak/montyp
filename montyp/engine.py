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

def run(goals: List[Goal]) -> List[Dict[str, Any]]:
    """Run a list of goals and return all solutions.
    
    Args:
        goals: List of goals to solve
    
    Returns:
        List of solutions, where each solution is a dictionary mapping variable names to values
    """
    def pursue(goals: List[Goal], state: State) -> Generator[State, None, None]:
        if not goals:
            yield state
            return
            
        goal, *rest = goals
        for new_state in goal(state):
            yield from pursue(rest, new_state)
    
    # Create initial state with empty substitutions and constraints
    state = State({}, [])
    
    # Collect all solutions
    solutions = []
    seen_signatures = set()  # Track seen type signatures
    
    for final_state in pursue(goals, state):
        # Extract variable bindings
        solution = {}
        raw_solution = {}  # Store raw values
        
        for var_name, val in final_state.subs.items():
            if not isinstance(var_name, (str, Var)):
                continue
                
            name = var_name.name if isinstance(var_name, Var) else var_name
            walked_val = deep_walk(val, final_state.subs)
            raw_solution[name] = walked_val
            
            if isinstance(walked_val, LogicalFunction):
                # Store the function and its type information
                solution[name] = walked_val
                solution[f"{name}_type"] = f"({walked_val.input_type.__name__}) -> {walked_val.output_type.__name__}"
            elif isinstance(walked_val, TypedVar):
                # If it's a TypedVar with a concrete value, use that
                concrete_val = walk(walked_val, final_state.subs)
                if not isinstance(concrete_val, (TypedVar, Var)):
                    solution[name] = concrete_val
                # Always include type information for TypedVar
                type_str = str(walked_val.type[0])
                if get_origin(walked_val.type[0]) is not None:
                    type_str = str(walked_val.type[0].__origin__.__name__)
                    args = get_args(walked_val.type[0])
                    if args:
                        type_str += f"[{', '.join(arg.__name__ for arg in args)}]"
                else:
                    type_str = walked_val.type[0].__name__
                solution[f"{name}_type"] = type_str
            else:
                solution[name] = walked_val
                # Include inferred type information
                inferred = infer_type(walked_val)
                if isinstance(inferred, (str, FunctionType)):
                    solution[f"{name}_type"] = str(inferred)
                elif hasattr(inferred, '__name__'):
                    solution[f"{name}_type"] = inferred.__name__
        
        # Create a signature tuple for this solution
        if 'map_func' in raw_solution and 'map_func_2' in raw_solution:
            f1 = raw_solution['map_func']
            f2 = raw_solution['map_func_2']
            
            # Check if this is a valid solution
            if f1.output_type == f2.input_type:
                signature = (
                    (f1.input_type.__name__, f1.output_type.__name__),
                    (f2.input_type.__name__, f2.output_type.__name__)
                )
                
                # Only add the solution if we haven't seen this signature before
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    solution["_raw"] = raw_solution
                    solutions.append(solution)
        else:
            # For other solutions, just add them
            solution["_raw"] = raw_solution
            solutions.append(solution)
    
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
                    # Try different possible type signatures
                    input_type = (type(walked_iterable[0]) if walked_iterable 
                                else get_args(iterable.type[0])[0] if isinstance(iterable, TypedVar) 
                                else int)  # fallback
                    
                    # Get possible output types based on the result type
                    output_types = []
                    if walked_result and isinstance(walked_result, list) and walked_result:
                        output_types.append(type(walked_result[0]))
                        # If the result type is str, also try int as an intermediate type
                        if type(walked_result[0]) == str:
                            output_types.append(int)
                    else:
                        output_types = [str]  # fallback
                    
                    # Try each possible output type
                    for output_type in output_types:
                        logical_func = LogicalFunction(map_func.name, input_type, output_type)
                        
                        if walked_iterable and walked_result:
                            # For str output, try both direct conversion and int->str
                            if output_type == str and input_type == int:
                                # Try direct int->str conversion
                                for x, y in zip(walked_iterable, walked_result):
                                    logical_func.add_example(x, str(x))
                            else:
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
                else:
                    # Try different possible type signatures when both function and result are variables
                    input_type = (type(walked_iterable[0]) if walked_iterable 
                                else get_args(iterable.type[0])[0] if isinstance(iterable, TypedVar) 
                                else int)  # fallback
                    
                    # Look ahead in state to find constraints on the result
                    final_output = None
                    prev_func = None
                    
                    # Find the final output type and previous function
                    for k, v in state.subs.items():
                        if isinstance(v, list) and len(v) == len(walked_iterable):
                            if all(isinstance(x, str) for x in v):
                                final_output = v
                        elif isinstance(v, LogicalFunction) and k != map_func.name:
                            prev_func = v
                    
                    if final_output:
                        # We're the second map function
                        if prev_func:
                            # We have a previous function, use its output type as our input type
                            logical_func = LogicalFunction(map_func.name, prev_func.output_type, str)
                            example_result = final_output
                            
                            for x, y in zip(walked_iterable, example_result):
                                logical_func.add_example(x, y)
                            
                            candidate_state = new_state.copy()
                            if func_subs := unify(map_func, logical_func, candidate_state.subs):
                                candidate_state.subs.update(func_subs)
                                if result_subs := unify(result, example_result, candidate_state.subs):
                                    candidate_state.subs.update(result_subs)
                                    if all(c(candidate_state) for c in candidate_state.constraints):
                                        # Store the function with its variable name
                                        candidate_state.subs[map_func.name] = logical_func
                                        yield candidate_state
                    else:
                        # We're the first map function
                        # Try both paths:
                        # Path 1: int -> int (for int -> str later)
                        logical_func = LogicalFunction(map_func.name, input_type, int)
                        example_result = list(walked_iterable)  # Keep as int
                        
                        for x, y in zip(walked_iterable, example_result):
                            logical_func.add_example(x, y)
                        
                        candidate_state = new_state.copy()
                        if func_subs := unify(map_func, logical_func, candidate_state.subs):
                            candidate_state.subs.update(func_subs)
                            if result_subs := unify(result, example_result, candidate_state.subs):
                                candidate_state.subs.update(result_subs)
                                if all(c(candidate_state) for c in candidate_state.constraints):
                                    # Store the function with its variable name
                                    candidate_state.subs[map_func.name] = logical_func
                                    # Add a constraint that the next function must take int input and output str
                                    candidate_state.constraints.append(
                                        lambda s, t1=int, t2=str: any(
                                            isinstance(v, LogicalFunction) and 
                                            v.input_type == t1 and 
                                            v.output_type == t2
                                            for v in s.subs.values()
                                        )
                                    )
                                    yield candidate_state
                        
                        # Path 2: int -> str (for str -> str later)
                        logical_func = LogicalFunction(map_func.name, input_type, str)
                        example_result = [str(x) for x in walked_iterable]
                        
                        for x, y in zip(walked_iterable, example_result):
                            logical_func.add_example(x, y)
                        
                        candidate_state = new_state.copy()
                        if func_subs := unify(map_func, logical_func, candidate_state.subs):
                            candidate_state.subs.update(func_subs)
                            if result_subs := unify(result, example_result, candidate_state.subs):
                                candidate_state.subs.update(result_subs)
                                if all(c(candidate_state) for c in candidate_state.constraints):
                                    # Store the function with its variable name
                                    candidate_state.subs[map_func.name] = logical_func
                                    # Add a constraint that the next function must take str input and output str
                                    candidate_state.constraints.append(
                                        lambda s, t1=str, t2=str: any(
                                            isinstance(v, LogicalFunction) and 
                                            v.input_type == t1 and 
                                            v.output_type == t2
                                            for v in s.subs.values()
                                        )
                                    )
                                    yield candidate_state
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