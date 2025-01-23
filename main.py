from montyp.engine import run, Var, applyo, ADD, MUL, REVERSE, Function

if __name__ == "__main__":
    # Test 1: x + 5 = 8
    x = Var()
    print("Test 1 Solutions:", run([
        applyo(ADD, x, 5, result=8)
    ]))

    # Test 2: (x + 5) * 2 = 16
    x, y = Var(), Var()
    print("\nTest 2 Solutions:", run([
        applyo(ADD, x, 5, result=y),
        applyo(MUL, y, 2, result=16)
    ]))

    # Test 3: Reverse string
    s = Var()
    print("\nTest 3 Solutions:", run([
        applyo(REVERSE, s, result="dlrow olleH")
    ]))
    
    # Test 4: Custom function with inverse
    from math import sqrt
    SQUARE = Function(lambda x: x * x, 
                     inverse=[(0, lambda res: sqrt(res) if res >= 0 else None)],
                     domain=lambda args: True)
    
    x = Var()
    print("\nTest 4 Solutions:", run([
        applyo(SQUARE, x, result=16)
    ]))