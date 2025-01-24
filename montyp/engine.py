from itertools import chain
from typing import Any, Dict, List, Optional, Generator, TypeVar, Set

from .schemas import Var, State, Goal

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

def run(goals: List[Goal], n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run a list of goals and return solutions.
    
    Args:
        goals: List of goals to satisfy
        n: Maximum number of solutions to return (None for all solutions)
    
    Returns:
        List of solutions, where each solution is a dictionary mapping
        variable names to their values
    """
    # Reset Var._id counter to ensure consistent behavior
    Var._id = 0
    
    # First, collect all variables from the goals
    original_vars = set()
    for goal in goals:
        # Extract the closure's variables from the goal function
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
            if not isinstance(val, Var):  # Only add if we have a concrete value
                sol[var.name] = val
        
        if sol:
            solutions.append(sol)
    
    return solutions