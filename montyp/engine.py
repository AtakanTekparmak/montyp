from itertools import chain
from typing import Any, Dict, List, Optional, Callable, Union, Generator, TypeVar, Iterable

from .schemas import Var, State, Goal
from .operations import Function

T = TypeVar('T')

def walk(var: Any, subs: Dict[Var, Any]) -> Any:
    """Walk through the substitution chain to find the final value.
    
    Args:
        var: The variable or value to resolve
        subs: Dictionary of variable substitutions
    
    Returns:
        The final value after following all substitutions
    """
    while isinstance(var, Var) and var in subs:
        var = subs[var]
    return var

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
    """Unify two terms, returning updated substitutions if successful.
    
    Args:
        u: First term to unify
        v: Second term to unify
        subs: Current substitution dictionary
    
    Returns:
        Updated substitutions if unification succeeds, None if it fails
    """
    u = walk(u, subs)
    v = walk(v, subs)
    
    if u == v:
        return subs
    if isinstance(u, Var):
        return {**subs, u: v}
    if isinstance(v, Var):
        return {**subs, v: u}
    
    if isinstance(u, (list, tuple)) and isinstance(v, (list, tuple)) and len(u) == len(v):
        new_subs = subs.copy()
        for ui, vi in zip(u, v):
            if (new_subs := unify(ui, vi, new_subs)) is None:
                return None
        return new_subs
    
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

def applyo(fn: Union[Function, Callable[..., Any]], *args: Any, result: Any) -> Goal:
    """Apply a function to arguments and unify with result.
    
    This function handles both:
        - Forward computation: compute result from arguments
        - Backward computation: find arguments that would give the result
    
    Args:
        fn: Function to apply (either a Function instance or regular callable)
        *args: Arguments to pass to the function
        result: Expected result to unify with
    
    Returns:
        A goal function that attempts to satisfy the function application
    """
    # If fn is not a Function instance, treat it as a regular forward-only function
    if not isinstance(fn, Function):
        fn = Function(fn)
    
    def goal(state: State) -> Generator[State, None, None]:
        resolved_args = [walk(arg, state.subs) for arg in args]
        resolved_result = walk(result, state.subs)
        
        # Forward computation: if we have all args, compute result
        if all(not isinstance(arg, Var) for arg in resolved_args):
            if fn.domain(resolved_args):
                try:
                    computed = fn(*resolved_args)
                    if new_subs := unify(computed, resolved_result, state.subs):
                        new_state = state.copy()
                        new_state.subs = new_subs
                        if all(c(new_state) for c in new_state.constraints):
                            yield new_state
                except:
                    pass
        
        # Backward computation: if we have the result and function has inverses
        elif not isinstance(resolved_result, Var) and fn.inverse:
            if isinstance(fn.inverse, list):
                # Try each inverse function for each variable argument
                for arg_idx, inverse_fn in fn.inverse:
                    if isinstance(resolved_args[arg_idx], Var):
                        other_args = [arg for i, arg in enumerate(resolved_args) if i != arg_idx]
                        if all(not isinstance(arg, Var) or arg in state.subs for arg in other_args):
                            try:
                                # Resolve any variables in other_args
                                resolved_other_args = [
                                    deep_walk(arg, state.subs) if isinstance(arg, Var) else arg 
                                    for arg in other_args
                                ]
                                # Apply inverse function to get the value for the variable
                                computed = inverse_fn(resolved_result, *resolved_other_args)
                                if computed is not None:
                                    if new_subs := unify(resolved_args[arg_idx], computed, state.subs):
                                        new_state = state.copy()
                                        new_state.subs = new_subs
                                        if all(c(new_state) for c in new_state.constraints):
                                            yield new_state
                            except:
                                pass
            else:
                # Single inverse function
                try:
                    computed = fn.inverse(resolved_result)
                    if new_subs := unify(resolved_args[0], computed, state.subs):
                        new_state = state.copy()
                        new_state.subs = new_subs
                        if all(c(new_state) for c in new_state.constraints):
                            yield new_state
                except:
                    pass
    return goal

def run(goals: List[Goal], n: Optional[int] = None) -> List[Dict[Var, Any]]:
    """Run a list of goals and return solutions.
    
    This function attempts to satisfy all goals in sequence, trying both
    forward and backward computation if needed.
    
    Args:
        goals: List of goals to satisfy
        n: Maximum number of solutions to return (None for all solutions)
    
    Returns:
        List of solutions, where each solution is a dictionary mapping
        variables to their values
    """
    states = [State({}, [])]
    
    # First pass: forward computation
    for goal in goals:
        states = list(chain.from_iterable(goal(s) for s in states))
        if not states:
            break
    
    # Second pass: backward computation if needed
    if not states and len(goals) > 1:
        states = [State({}, [])]
        for goal in reversed(goals):
            states = list(chain.from_iterable(goal(s) for s in states))
            if not states:
                break
    
    solutions = []
    for state in states[:n]:
        sol = {}
        # First collect all variables from the substitution chain
        vars_to_resolve = set(state.subs.keys())
        for val in state.subs.values():
            if isinstance(val, Var):
                vars_to_resolve.add(val)
        
        # Then resolve each variable
        for var in vars_to_resolve:
            val = deep_walk(var, state.subs)
            if not isinstance(val, Var):
                sol[var] = val
        if sol:
            solutions.append(sol)
    
    return solutions