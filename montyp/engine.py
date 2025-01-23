from itertools import chain
from typing import Any, Dict, List, Optional, Callable, Generator, Iterable

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

def deep_walk(val: Any, subs: Dict[Var, Any]) -> Any:
    """Recursively resolve variables in nested structures"""
    val = walk(val, subs)
    if isinstance(val, (list, tuple)):
        return type(val)(deep_walk(v, subs) for v in val)
    if isinstance(val, dict):
        return {k: deep_walk(v, subs) for k, v in val.items()}
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
        new_subs = subs
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

def conj(goals: List[Goal]) -> Goal:
    def goal(state: State):
        def apply_goals(states: Iterable[State], goals: List[Goal]):
            if not goals:
                return states
            current, *rest = goals
            new_states = chain.from_iterable(current(s) for s in states)
            return apply_goals(new_states, rest)
        return apply_goals([state], goals)
    return goal

def conde(*clauses: List[Goal]) -> Goal:
    def goal(state: State):
        return chain.from_iterable(
            conj(clause)(state) 
            for clause in clauses
        )
    return goal

def fresh(f: Callable[[Var], Goal]) -> Goal:
    def goal(state: State):
        return f(Var())(state)
    return goal

def constraint(pred: Callable[[State], bool]) -> Goal:
    def goal(state: State):
        if pred(state):
            yield state
    return goal

def run(goals: List[Goal], n: Optional[int] = None) -> List[Dict[Var, Any]]:
    states = [State({}, [])]
    
    for goal in goals:
        states = list(chain.from_iterable(goal(s) for s in states))
        if not states:
            break
    
    solutions = []
    for state in states[:n]:
        sol = {}
        for var in state.subs:
            val = deep_walk(var, state.subs)
            sol[var] = val if not isinstance(val, Var) else '?'
        solutions.append(sol)
    
    return solutions

def typeo(var: Var, expected_type: str) -> Goal:
    def check(state: State):
        value = deep_walk(var, state.subs)
        return isinstance(value, Var) or type(value).__name__ == expected_type
    return constraint(check)