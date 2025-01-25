import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar
from typing import List

def main():
    # Define functions
    def double(x: int) -> int:
        return x * 2
        
    def to_string(x: int) -> str:
        return str(x)
    
    # Define variables
    numbers = TypedVar('numbers', List[int])
    doubled = Var('doubled')
    strings = Var('strings')
    numbers_type = Var('numbers_type')
    doubled_type = Var('doubled_type')
    strings_type = Var('strings_type')

    # Define constraints
    result = run([
        eq(numbers, [1, 2, 3, 4]),
        type_of(numbers, numbers_type),
        apply(map, [double, numbers], doubled),
        type_of(doubled, doubled_type),
        apply(map, [to_string, doubled], strings),
        type_of(strings, strings_type)
    ])
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 