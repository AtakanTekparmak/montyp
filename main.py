from montyp.engine import run, eq, type_of, getitem
from montyp.schemas import Var, TypedVar
from typing import List, Union, Tuple, Set, Dict

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
        type_of(a, t)
    ]))
    
    # Test 3: Generic type constraints with variable propagation
    list_of_ints = TypedVar('nums', List[int])
    elem = TypedVar('elem', int)
    unknown = Var('unknown')  # This should get the int constraint
    
    print("\nTest Generic Constraints:", run([
        eq(list_of_ints, [1, 2, unknown]),  # Changed order to help unification
        eq(unknown, 3),
        eq(elem, unknown)
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
    result = Var('result')
    
    print("\nTest Nested Types:", run([
        eq(nested, [[1, 2], [3, 4]]),
        eq(result, nested)  # result should get the nested type constraint
    ]))
    
    # Test 6: Complex type inference with dictionaries
    dict_var = Var('dict')
    dict_type = Var('dict_type')
    
    print("\nTest Dict Types:", run([
        eq(dict_var, {'a': [1, 2], 'b': [3, 4]}),
        type_of(dict_var, dict_type)
    ]))
    
    # Test 7: Type constraint propagation
    int_var = TypedVar('int_var', int)
    a = Var('a')
    b = Var('b')
    
    print("\nTest Constraint Propagation:", run([
        eq(int_var, a),  # a gets int constraint
        eq(a, b),        # b should also get int constraint
        eq(b, 42)        # This should work
    ]))
    
    # Test 8: Complex nested structures with type constraints
    matrix = TypedVar('matrix', List[List[Union[int, float]]])
    row = Var('row')
    element = Var('element')
    
    print("\nTest Matrix Operations:", run([
        eq(matrix, [[1, 2.5], [3, 4.0]]),
        getitem(matrix, 0, row),      # row should be [1, 2.5]
        getitem(row, 1, element)      # element should be 2.5
    ]))
    
    # Test 9: Type inference with sets and tuples
    container = Var('container')
    container_type = Var('container_type')
    
    print("\nTest Container Types:", run([
        eq(container, {(1, 'a'), (2, 'b'), (3, 'c')}),
        type_of(container, container_type)
    ]))
    
    # Test 10: Recursive type constraints
    tree = TypedVar('tree', Dict[str, Union[int, List[int]]])
    value = Var('value')
    
    print("\nTest Recursive Types:", run([
        eq(tree, {'root': 1, 'children': [2, 3, 4]}),
        getitem(tree, 'children', value),  # get the children list
        getitem(value, 1, element)         # get element at index 1 (should be 3)
    ])) 