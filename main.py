import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar, MontyEncoder
from typing import List

def main():
    # Define variables
    numbers = TypedVar('numbers', List[int])
    map_func = Var('map_func')
    map_func_2 = Var('map_func_2')
    numbers_2 = Var('numbers_2')
    
    # Define constraints
    solutions = run([
        eq(numbers, [1, 2, 3, 4]),
        apply(map, [map_func, numbers], numbers_2),
        apply(map, [map_func_2, numbers_2], ["2", "4", "6", "8"]),
    ])
    
    # Remove raw values and redundant type information
    clean_solutions = []
    for sol in solutions:
        clean_sol = {}
        for k, v in sol.items():
            if k == '_raw':
                continue
            if k.endswith('_type') and k.startswith('map_func'):
                continue
            clean_sol[k] = v
        clean_solutions.append(clean_sol)
    
    print(json.dumps(clean_solutions, indent=2, cls=MontyEncoder))

if __name__ == "__main__":
    main() 