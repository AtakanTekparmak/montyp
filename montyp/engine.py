from itertools import chain
from typing import Any, Dict, List, Optional, Callable, Generator, Iterable, Union, Tuple
from operator import add, mul
from inspect import signature
import math

def string_reverse(s):
    return s[::-1]

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

Goal = Callable[[State], Generator[State, None, None]]

def walk(var: Any, subs: Dict[Var, Any]) -> Any:
    while isinstance(var, Var) and var in subs:
        var = subs[var]
    return var

def deep_walk(val: Any, subs: Dict[Var, Any], depth: int = 0) -> Any:
    if depth > 100:  # Prevent infinite recursion
        return val
    
    val = walk(val, subs)
    
    if isinstance(val, (list, tuple)):
        return type(val)(deep_walk(v, subs, depth + 1) for v in val)
    
    return val

def unify(u: Any, v: Any, subs: Dict[Var, Any]) -> Optional[Dict[Var, Any]]:
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
    def goal(state: State):
        if new_subs := unify(a, b, state.subs):
            new_state = state.copy()
            new_state.subs = new_subs
            if all(c(new_state) for c in new_state.constraints):
                yield new_state
    return goal

class Function:
    """Wrapper for functions with additional properties for logical programming."""
    def __init__(self, fn: Callable, 
                 inverse: Optional[Union[Callable, List[Tuple[int, Callable]]]] = None,
                 domain: Optional[Callable[[List[Any]], bool]] = None):
        self.fn = fn
        self.inverse = inverse  # Either a single inverse function or list of (arg_index, inverse_fn)
        self.domain = domain or (lambda args: True)  # Domain constraint
        self.sig = signature(fn)
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

# Define some common functions with their inverses
ADD = Function(add, 
              inverse=[(0, lambda res, y: res - y),   # x + y = res -> x = res - y
                      (1, lambda res, x: res - x)])   # x + y = res -> y = res - x

MUL = Function(mul,
              inverse=[(0, lambda res, y: res / y if y != 0 else None),  # x * y = res -> x = res / y
                      (1, lambda res, x: res / x if x != 0 else None)],  # x * y = res -> y = res / x
              domain=lambda args: 0 not in args[1:])  # Prevent division by zero

REVERSE = Function(string_reverse,
                  inverse=[(0, string_reverse)])  # reverse(x) = y -> x = reverse(y)

def applyo(fn: Union[Function, Callable], *args: Any, result: Any) -> Goal:
    # If fn is not a Function instance, treat it as a regular forward-only function
    if not isinstance(fn, Function):
        fn = Function(fn)
    
    def goal(state: State):
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