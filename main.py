import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar
from typing import List

def main():
    # Define variables
    numbers = TypedVar('numbers', List[int])
    map_func = Var('map_func')
    map_func_2 = Var('map_func_2')
    
    # Define constraints
    solutions = run([
        eq(numbers, [1, 2, 3, 4]),
        apply(map, [map_func, numbers], [2, 4, 6, 8]),
        apply(map, [map_func_2, numbers], ["2", "4", "6", "8"]),
    ])
    
    # Remove raw values before JSON serialization
    clean_solutions = [{k: v for k, v in sol.items() if k != '_raw'} for sol in solutions]
    print(json.dumps(clean_solutions, indent=2))

if __name__ == "__main__":
    main() 