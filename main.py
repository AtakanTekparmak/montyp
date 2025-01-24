from montyp.engine import run, applyo, eq
from montyp.schemas import Var
from montyp.operations import ADD, MUL, REVERSE, SQUARE, SUB, Function
from operator import truediv
from math import sqrt, pow

if __name__ == "__main__":
    # Test 1: Solve the equation x + 5 = 8
    # Expected: x = 3
    x = Var()
    print("Test 1 Solutions:", run([
        applyo(ADD, x, 5, result=8)
    ]))

    # Test 2: Solve the system of equations:
    # y = x + 5
    # 16 = y * 2
    # Expected: x = 3, y = 8
    x, y = Var(), Var()
    print("\nTest 2 Solutions:", run([
        applyo(ADD, x, 5, result=y),
        applyo(MUL, y, 2, result=16)
    ]))

    # Test 3: Find a string that when reversed equals "dlrow olleH"
    # Expected: s = "Hello world"
    s = Var()
    print("\nTest 3 Solutions:", run([
        applyo(REVERSE, s, result="dlrow olleH")
    ]))

    # Test 4: Find a number that when squared equals 16
    # Expected: x = 4 (or -4, but our inverse function only returns the positive root)
    x = Var()
    print("\nTest 4 Solutions:", run([
        applyo(SQUARE, x, result=16)
    ]))

    # Test 5: Direct unification of variables
    # This shows how variables can be unified with values and other variables
    x, y, z = Var(), Var(), Var()
    print("\nTest 5 Solutions:", run([
        eq(x, 5),           # x = 5
        eq(y, x),          # y = x (so y = 5)
        eq(z, [x, y, 42])  # z = [5, 5, 42]
    ]))

    # Test 6: Complex unification with nested structures
    # This demonstrates unifying variables within lists and tuples
    a, b, c = Var(), Var(), Var()
    print("\nTest 6 Solutions:", run([
        eq([1, b, 3], [a, 2, c]),  # Unifies a=1, b=2, c=3
        eq((a, b), (1, 2))         # Confirms a=1, b=2
    ]))

    # Test 7: Mixing eq and applyo
    # Shows how to combine direct unification with function application
    x, y, result = Var(), Var(), Var()
    print("\nTest 7 Solutions:", run([
        eq(x, 10),                    # x = 10
        applyo(ADD, x, 5, result=y),  # y = x + 5 = 15
        eq([x, y], result)            # result = [10, 15]
    ]))

    # Test 8: Custom function with inverse (subtraction)
    a, b = Var(), Var()
    print("\nTest 8 Solutions:", run([
        applyo(SUB, a, 5, result=3),    # Using SUB instead of sub
        applyo(SUB, 10, b, result=4)
    ]))

    # Test 9: Custom function with manual inverse
    def double(x):
        return x * 2
    
    # Create a logical function with inverse
    DOUBLE = Function(
        double,
        inverse=[(0, lambda res: res / 2)],  # Add inverse
        domain=lambda args: True
    )
    
    x = Var()
    print("\nTest 9 Solutions:", run([
        applyo(DOUBLE, x, result=8)  # Now works bidirectionally
    ]))

