import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar, MontyEncoder
from typing import List
from functools import reduce

def main():
    # Example 1: Map pipeline
    numbers = TypedVar('numbers', List[int])
    map_func = Var('map_func')
    map_func_2 = Var('map_func_2')
    numbers_2 = Var('numbers_2')
    
    solutions = run([
        eq(numbers, [1, 2, 3, 4]),
        apply(map, [map_func, numbers], numbers_2),
        apply(map, [map_func_2, numbers_2], ["2", "4", "6", "8"]),
    ])
    
    print("Map pipeline solutions:")
    print(json.dumps(solutions, indent=2, cls=MontyEncoder))
    
    # Example 2: Filter-Map-Reduce pipeline
    def is_even(x: int) -> bool:
        return x % 2 == 0
        
    numbers = TypedVar('numbers', List[int])
    transform = Var('transform')
    filtered = Var('filtered')
    mapped = Var('mapped')
    total = Var('total')
    
    solutions = run([
        eq(numbers, [1, 2, 3, 4, 5, 6]),
        apply(filter, [is_even, numbers], filtered),
        apply(map, [transform, filtered], mapped),
        eq(mapped, [4, 8, 12]),  # We want to double the even numbers
        apply(reduce, [lambda x, y: x + y, mapped], total)
    ])
    
    print("\nFilter-Map-Reduce pipeline solutions:")
    print(json.dumps(solutions, indent=2, cls=MontyEncoder))

if __name__ == "__main__":
    main() 