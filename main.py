from montyp.engine import run, applyo
from montyp.schemas import Var
from montyp.operations import ADD, MUL, REVERSE, SQUARE

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