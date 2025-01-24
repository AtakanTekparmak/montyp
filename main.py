from montyp.engine import run, eq
from montyp.schemas import Var

if __name__ == "__main__":
    # Test : Direct unification of variables
    # This shows how variables can be unified with values and other variables
    x, y, z = Var('x'), Var('y'), Var('z')
    print("\nTest 5 Solutions:", run([
        eq(x, 5),           # x = 5
        eq(y, x),          # y = x (so y = 5)
        eq(z, [x, y, 42])  # z = [5, 5, 42]
    ]))

    # Test 2: Complex unification with nested structures
    # This demonstrates unifying variables within lists and tuples
    a, b, c = Var('a'), Var('b'), Var('c')
    print("\nTest 6 Solutions:", run([
        eq([1, b, 3], [a, 2, c]),  # Unifies a=1, b=2, c=3
        eq((a, b), (1, 2))         # Confirms a=1, b=2
    ]))
