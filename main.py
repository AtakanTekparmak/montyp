from montyp.engine import run, eq, type_of, getitem
from montyp.schemas import Var, TypedVar
from typing import List, Union, Dict

def main():
    # Simple demonstration
    x = TypedVar('x', int)
    y = Var('y')
    z = Var('z')
    
    result = run([
        eq(x, 42),
        eq(y, "Hello"),
        type_of(y, Var('y_type')),
        eq(z, [x, 43]),
        type_of(z, Var('z_type'))
    ])
    
    print("Demo result:", result)

if __name__ == "__main__":
    main() 