import json
from montyp.engine import run, eq, type_of
from montyp.schemas import Var, TypedVar

def main():
    # Define functions
    def add(x: int, y: int) -> int:
        return x + y
    
    # Define variables
    x = TypedVar('x', int)
    y = Var('y')
    z = Var('z')
    x_type = Var('x_type')
    add_type = Var('add_type')

    # Define constraints
    result = run([
        eq(x, 42),
        eq(y, "Hello"),
        eq(z, (x, y)),
        type_of(x, x_type),
        type_of(add, add_type)
    ])
    
    print("Demo result:\n", json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 