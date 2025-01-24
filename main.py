from montyp.engine import run, eq, type_of
from montyp.schemas import Var, TypedVar
from typing import List, Union, Tuple

if __name__ == "__main__":
    # Test 1: Basic type constraints
    x = TypedVar('x', int)
    y = TypedVar('y', (int, float))  # Union type
    z = TypedVar('z', str, nullable=True)  # Nullable type
    
    print("\nTest Type Constraints:", run([
        eq(x, 42),          # OK - int
        eq(y, 3.14),        # OK - float
        eq(z, None),        # OK - nullable
    ]))
    
    # Test 2: Type inference with explicit type variable
    a = Var('a')
    t = Var('type_of_a')
    
    print("\nTest Type Inference:", run([
        eq(a, [1, 2, 3]),
        type_of(a, t)  # Now properly connects inferred type to t
    ]))
    
    # Test 3: Generic type constraints
    list_of_ints = TypedVar('nums', List[int])  # Now supports generic types
    elem = TypedVar('elem', int)
    
    print("\nTest Generic Constraints:", run([
        eq(list_of_ints, [1, 2, elem]),
        eq(elem, 3)
    ]))
    
    # Test 4: Mixed type inference
    mixed = Var('mixed')
    mixed_type = Var('mixed_type')
    
    print("\nTest Mixed Types:", run([
        eq(mixed, [1, "two", 3.0]),
        type_of(mixed, mixed_type)
    ]))
    
    # Test 5: Nested type constraints
    nested = TypedVar('nested', List[List[int]])
    
    print("\nTest Nested Types:", run([
        eq(nested, [[1, 2], [3, 4]]),  # Should work
        eq(nested, [[1, 2], [3, "4"]])  # Should fail
    ]))
    
    # Test 6: Complex type inference
    complex_var = Var('complex')
    complex_type = Var('complex_type')
    
    print("\nTest Complex Types:", run([
        eq(complex_var, [(1, 2), [3, 4], {5, 6}]),
        type_of(complex_var, complex_type)
    ]))