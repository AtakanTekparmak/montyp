import unittest
import sys
from pathlib import Path

# Add the parent directory to Python path for relative imports
sys.path.append(str(Path(__file__).parent.parent))

from montyp.engine import run, eq, type_of, getitem
from montyp.schemas import Var, TypedVar
from typing import List, Union, Dict

class TestMontyEngine(unittest.TestCase):
    def test_basic_type_constraints(self):
        """Test basic type constraints with int, float and nullable types"""
        x = TypedVar('x', int)
        y = TypedVar('y', (int, float))  # Union type
        z = TypedVar('z', str, nullable=True)  # Nullable type
        
        result = run([
            eq(x, 42),          # OK - int
            eq(y, 3.14),        # OK - float
            eq(z, None),        # OK - nullable
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['x'], 42)
        self.assertEqual(result[0]['y'], 3.14)
        self.assertIsNone(result[0]['z'])

    def test_type_inference(self):
        """Test type inference with explicit type variable"""
        a = Var('a')
        t = Var('type_of_a')
        
        result = run([
            eq(a, [1, 2, 3]),
            type_of(a, t)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['a'], [1, 2, 3])
        self.assertEqual(result[0]['type_of_a'], 'List[int]')

    def test_generic_constraints(self):
        """Test generic type constraints with variable propagation"""
        list_of_ints = TypedVar('nums', List[int])
        elem = TypedVar('elem', int)
        unknown = Var('unknown')
        type_of_unknown = Var('type_of_unknown')

        result = run([
            eq(list_of_ints, [1, 2, unknown]),  # This should constrain unknown to be an int
            eq(unknown, 3),                     # This should work because 3 is an int
            eq(elem, unknown),                  # This should work because unknown is an int
            type_of(unknown, type_of_unknown)   # This should show int
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['nums'], [1, 2, 3])
        self.assertEqual(result[0]['unknown'], 3)
        self.assertEqual(result[0]['elem'], 3)
        self.assertEqual(result[0]['type_of_unknown'], 'int')

    def test_mixed_types(self):
        """Test mixed type inference"""
        mixed = Var('mixed')
        mixed_type = Var('mixed_type')
        
        result = run([
            eq(mixed, [1, "two", 3.0]),
            type_of(mixed, mixed_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['mixed'], [1, "two", 3.0])
        self.assertEqual(result[0]['mixed_type'], 'List[Union[float, int, str]]')

    def test_nested_types(self):
        """Test nested type constraints"""
        nested = TypedVar('nested', List[List[int]])
        result_var = Var('result')
        type_of_result = Var('type_of_result')
        
        result = run([
            eq(nested, [[1, 2], [3, 4]]),
            eq(result_var, nested),  # result should get the nested type constraint
            type_of(result_var, type_of_result)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['nested'], [[1, 2], [3, 4]])
        self.assertEqual(result[0]['result'], [[1, 2], [3, 4]])
        self.assertEqual(result[0]['type_of_result'], 'List[List[int]]')

    def test_dict_types(self):
        """Test dictionary type inference"""
        dict_var = Var('dict')
        dict_type = Var('dict_type')
        
        result = run([
            eq(dict_var, {'a': [1, 2], 'b': [3, 4]}),
            type_of(dict_var, dict_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['dict'], {'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(result[0]['dict_type'], 'Dict[str, List[int]]')

    def test_constraint_propagation(self):
        """Test type constraint propagation"""
        int_var = TypedVar('int_var', int)
        a = Var('a')
        b = Var('b')
        
        result = run([
            eq(int_var, a),  # a gets int constraint
            eq(a, b),        # b should also get int constraint
            eq(b, 42)        # This should work
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['b'], 42)
        self.assertEqual(result[0]['a_type'], 'int')
        self.assertEqual(result[0]['int_var_type'], 'int')

    def test_matrix_operations(self):
        """Test matrix operations with type constraints"""
        matrix = TypedVar('matrix', List[List[Union[int, float]]])
        row = Var('row')
        element = Var('element')
        
        result = run([
            eq(matrix, [[1, 2.5], [3, 4.0]]),
            getitem(matrix, 0, row),      # row should be [1, 2.5]
            getitem(row, 1, element)      # element should be 2.5
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['matrix'], [[1, 2.5], [3, 4.0]])
        self.assertEqual(result[0]['element'], 2.5)
        self.assertEqual(result[0]['row'], [1, 2.5])

    def test_container_types(self):
        """Test type inference with sets and tuples"""
        container = Var('container')
        container_type = Var('container_type')
        
        result = run([
            eq(container, {(1, 'a'), (2, 'b'), (3, 'c')}),
            type_of(container, container_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['container'], {(1, 'a'), (2, 'b'), (3, 'c')})
        self.assertEqual(result[0]['container_type'], 'Set[Tuple[Union[int, str], ...]]')

    def test_recursive_types(self):
        """Test recursive type constraints"""
        tree = TypedVar('tree', Dict[str, Union[int, List[int]]])
        value = Var('value')
        element = Var('element')
        
        result = run([
            eq(tree, {'root': 1, 'children': [2, 3, 4]}),
            getitem(tree, 'children', value),  # get the children list
            getitem(value, 1, element)         # get element at index 1 (should be 3)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['tree'], {'root': 1, 'children': [2, 3, 4]})
        self.assertEqual(result[0]['element'], 3)
        self.assertEqual(result[0]['value'], [2, 3, 4])

if __name__ == '__main__':
    unittest.main() 