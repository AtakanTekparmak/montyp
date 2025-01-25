import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar

def main():
    # Define functions
    def add(x: int, y: int) -> int:
        return x + y
    
    # Define variables
    x = TypedVar('x', int)
    y = Var('y')  # Untyped variable - will be deduced
    z = Var('z')

    x_type = Var('x_type')
    add_type = Var('add_type')
    y_type = Var('y_type')

    # Define constraints
    result = run([
        eq(x, 42),
        eq(z, (x, y)),
        type_of(x, x_type),
        type_of(add, add_type),    # We know we want k to be 52
        apply(add, [x, y], 52),  # This will deduce y = 10 since 42 + 10 = 52
        type_of(y, y_type)
    ])
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 