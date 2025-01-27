import unittest
import sys
from pathlib import Path
from functools import reduce
from typing import List, Union, Dict, get_type_hints, Any

# Add the parent directory to Python path for relative imports
sys.path.append(str(Path(__file__).parent.parent))

from montyp.engine import run, eq, type_of, getitem, apply
from montyp.schemas import Var, TypedVar, FunctionType

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
        self.assertEqual(result[0]['container_type'], 'Set[Tuple[int, str]]')

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

    def test_typed_var_in_list(self):
        """Test type inference with TypedVars inside lists"""
        x = TypedVar('x', int)
        z = Var('z')
        z_type = Var('z_type')
        
        result = run([
            eq(x, 42),
            eq(z, [x, 43]),
            type_of(z, z_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['z'], [42, 43])
        self.assertEqual(result[0]['z_type'], 'List[int]')

    def test_function_types(self):
        """Test function type inference and constraints"""
        def add(x: int, y: int) -> int:
            return x + y
        
        f = Var('f')
        f_type = Var('f_type')
        
        result = run([
            eq(f, add),
            type_of(f, f_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['f'], add)
        self.assertEqual(str(result[0]['f_type']), '(int, int) -> int')

    def test_function_type_constraints(self):
        """Test function type constraints"""
        def add(x: int, y: int) -> int:
            return x + y
        
        f = TypedVar('f', FunctionType([int, int], int))
        
        result = run([
            eq(f, add)  # This should work
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['f'], add)
        
        # Try with incompatible function
        def concat(x: str, y: str) -> str:
            return x + y
        
        result = run([
            eq(f, concat)  # This should fail
        ])
        
        self.assertEqual(len(result), 0)

    def test_nested_function_types(self):
        """Test nested function types in data structures"""
        def inc(x: int) -> int:
            return x + 1
        
        def double(x: int) -> int:
            return x * 2
        
        funcs = Var('funcs')
        funcs_type = Var('funcs_type')
        
        result = run([
            eq(funcs, [inc, double]),  # Use named function instead of lambda
            type_of(funcs, funcs_type)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['funcs_type'], 'List[(int) -> int]')

class TestFunctionApplication(unittest.TestCase):
    """Test cases for function application constraints"""
    
    def test_basic_function_application(self):
        """Test basic function application with simple types"""
        def add(x: int, y: int) -> int:
            return x + y
            
        x = TypedVar('x', int)
        y = TypedVar('y', int)
        result = Var('result')
        
        solutions = run([
            eq(x, 42),
            eq(y, 10),
            apply(add, [x, y], result)
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['result'], 52)
    
    def test_function_application_with_type_mismatch(self):
        """Test function application with incompatible types"""
        def add(x: int, y: int) -> int:
            return x + y
            
        x = TypedVar('x', int)
        y = TypedVar('y', str)  # Wrong type
        result = Var('result')
        
        solutions = run([
            eq(x, 42),
            eq(y, "hello"),
            apply(add, [x, y], result)
        ])
        
        # Should fail due to type mismatch
        self.assertEqual(len(solutions), 0)
    
    def test_nested_function_application(self):
        """Test nested function applications"""
        def add(x: int, y: int) -> int:
            return x + y
            
        def multiply(x: int, y: int) -> int:
            return x * y
            
        x = TypedVar('x', int)
        y = TypedVar('y', int)
        z = TypedVar('z', int)
        temp = Var('temp')
        result = Var('result')
        
        solutions = run([
            eq(x, 2),
            eq(y, 3),
            eq(z, 4),
            apply(add, [x, y], temp),      # 2 + 3 = 5
            apply(multiply, [temp, z], result)  # 5 * 4 = 20
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['temp'], 5)
        self.assertEqual(solutions[0]['result'], 20)
    
    def test_function_application_with_variables(self):
        """Test function application where some arguments are variables"""
        def add(x: int, y: int) -> int:
            return x + y
            
        x = TypedVar('x', int)
        y = Var('y')  # Untyped variable
        result = Var('result')
        
        solutions = run([
            eq(x, 42),
            eq(result, 52),  # We know the result we want
            apply(add, [x, y], result)  # Should deduce y = 10
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['y'], 10)
        self.assertEqual(solutions[0]['result'], 52)

class TestHigherOrderFunctions(unittest.TestCase):
    """Test cases for higher-order function applications"""
    
    def test_map_with_simple_function(self):
        """Test map function with a simple transformation"""
        def double(x: int) -> int:
            return x * 2
            
        input_list = TypedVar('input', List[int])
        result = Var('result')
        result_type = Var('result_type')
        
        solutions = run([
            eq(input_list, [1, 2, 3]),
            apply(map, [double, input_list], result),
            type_of(result, result_type)
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['result'], [2, 4, 6])
        self.assertEqual(solutions[0]['result_type'], 'List[int]')
    
    def test_map_with_type_inference(self):
        """Test map with type inference of input and output lists"""
        def to_string(x: int) -> str:
            return str(x)
            
        input_list = Var('input')  # Untyped variable
        result = Var('result')
        input_type = Var('input_type')
        result_type = Var('result_type')
        
        solutions = run([
            eq(input_list, [1, 2, 3]),
            type_of(input_list, input_type),
            apply(map, [to_string, input_list], result),
            type_of(result, result_type)
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['input_type'], 'List[int]')
        self.assertEqual(solutions[0]['result_type'], 'List[str]')
        self.assertEqual(solutions[0]['result'], ['1', '2', '3'])
    
    def test_map_with_deduced_function(self):
        """Test map where we deduce the transformation function"""
        def double(x: int) -> int:
            return x * 2
            
        def triple(x: int) -> int:
            return x * 3
            
        input_list = TypedVar('input', List[int])
        transform = Var('transform')
        result = Var('result')
        f1 = Var('f1')
        f2 = Var('f2')
        
        solutions = run([
            eq(f1, double),
            eq(f2, triple),
            eq(input_list, [1, 2, 3]),
            eq(result, [2, 4, 6]),
            apply(map, [transform, input_list], result),
            type_of(transform, FunctionType([int], int))
        ])
        
        self.assertEqual(len(solutions), 1)
        # Use the raw value for function testing
        raw_transform = solutions[0]['_raw']['transform']
        self.assertEqual(raw_transform(2), 4)

    def test_nested_higher_order_functions(self):
        """Test composition of higher-order functions"""
        def increment(x: int) -> int:
            return x + 1
            
        def stringify(x: int) -> str:
            return str(x)
            
        input_list = TypedVar('input', List[int])
        temp_result = Var('temp_result')
        final_result = Var('final_result')
        
        solutions = run([
            eq(input_list, [1, 2, 3]),
            apply(map, [increment, input_list], temp_result),
            apply(map, [stringify, temp_result], final_result)
        ])
        
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0]['temp_result'], [2, 3, 4])
        self.assertEqual(solutions[0]['final_result'], ['2', '3', '4'])

    def test_map_type_signatures(self):
        """Test finding multiple possible type signatures for map functions"""
        map_func = Var('map_func')
        map_func_2 = Var('map_func_2')
        input_list = [1, 2, 3]
        intermediate = Var('intermediate')
        final = ["1", "2", "3"]

        # Find possible type signatures where:
        # Case 1: map_func: (int) -> int, map_func_2: (int) -> str
        # Case 2: map_func: (int) -> str, map_func_2: (str) -> str
        result = run([
            apply(map, [map_func, input_list], intermediate),
            apply(map, [map_func_2, intermediate], final)
        ])

        self.assertEqual(len(result), 2)
        
        # Check both solutions are found
        solutions = []
        for r in result:
            f1 = r['map_func']
            f2 = r['map_func_2']
            solutions.append((
                (f1.input_type.__name__, f1.output_type.__name__),
                (f2.input_type.__name__, f2.output_type.__name__)
            ))

        expected_solutions = [
            (('int', 'int'), ('int', 'str')),
            (('int', 'str'), ('str', 'str'))
        ]

        self.assertEqual(sorted(solutions), sorted(expected_solutions))

    def test_filter_function(self):
        """Test filter higher-order function with type inference"""
        def is_even(x: int) -> bool:
            return x % 2 == 0
            
        numbers = TypedVar('numbers', List[int])
        filter_func = Var('filter_func')
        filtered = Var('filtered')
        
        result = run([
            eq(numbers, [1, 2, 3, 4, 5, 6]),
            eq(filter_func, is_even),
            apply(filter, [filter_func, numbers], filtered)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['filtered'], [2, 4, 6])
        self.assertEqual(result[0]['filtered_type'], 'List[int]')
        
    def test_reduce_function(self):
        """Test reduce higher-order function with type inference"""
        def sum_func(x: int, y: int) -> int:
            return x + y
            
        numbers = TypedVar('numbers', List[int])
        reduce_func = Var('reduce_func')
        reduced = Var('reduced')
        
        result = run([
            eq(numbers, [1, 2, 3, 4]),
            eq(reduce_func, sum_func),
            apply(reduce, [reduce_func, numbers], reduced)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['reduced'], 10)
        self.assertEqual(result[0]['reduced_type'], 'int')
        
    def test_reduce_with_initial(self):
        """Test reduce with initial value"""
        def concat(acc: str, x: int) -> str:
            return f"{acc},{x}"
            
        numbers = TypedVar('numbers', List[int])
        reduce_func = Var('reduce_func')
        reduced = Var('reduced')
        
        result = run([
            eq(numbers, [1, 2, 3]),
            eq(reduce_func, concat),
            apply(reduce, [reduce_func, numbers, "start"], reduced)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['reduced'], "start,1,2,3")
        self.assertEqual(result[0]['reduced_type'], 'str')
        
    def test_composed_higher_order_functions(self):
        """Test composition of multiple higher-order functions"""
        def double(x: int) -> int:
            return x * 2
            
        def is_greater_than_five(x: int) -> bool:
            return x > 5
            
        def sum_func(x: int, y: int) -> int:
            return x + y
            
        numbers = TypedVar('numbers', List[int])
        doubled = Var('doubled')
        filtered = Var('filtered')
        final_sum = Var('final_sum')
        
        result = run([
            eq(numbers, [1, 2, 3, 4]),
            apply(map, [double, numbers], doubled),
            apply(filter, [is_greater_than_five, doubled], filtered),
            apply(reduce, [sum_func, filtered], final_sum)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['doubled'], [2, 4, 6, 8])
        self.assertEqual(result[0]['filtered'], [6, 8])
        self.assertEqual(result[0]['final_sum'], 14)

    def test_composed_higher_order_functions_with_filter(self):
        """Test composition of map and filter"""
        def double(x: int) -> int:
            return x * 2
            
        def is_even(x: int) -> bool:
            return x % 2 == 0
            
        numbers = TypedVar('numbers', List[int])
        doubled = Var('doubled')
        filtered = Var('filtered')
        
        result = run([
            eq(numbers, [1, 2, 3, 4]),
            apply(map, [double, numbers], doubled),
            apply(filter, [is_even, doubled], filtered)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['doubled'], [2, 4, 6, 8])
        self.assertEqual(result[0]['filtered'], [2, 4, 6, 8])

    def test_reduce_with_map(self):
        """Test composition of map and reduce"""
        def double(x: int) -> int:
            return x * 2
            
        def sum_func(x: int, y: int) -> int:
            return x + y
            
        numbers = TypedVar('numbers', List[int])
        doubled = Var('doubled')
        total = Var('total')
        
        result = run([
            eq(numbers, [1, 2, 3, 4]),
            apply(map, [double, numbers], doubled),
            apply(reduce, [sum_func, doubled], total)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['doubled'], [2, 4, 6, 8])
        self.assertEqual(result[0]['total'], 20)

    def test_filter_map_reduce(self):
        """Test composition of filter, map, and reduce"""
        def is_even(x: int) -> bool:
            return x % 2 == 0
            
        def triple(x: int) -> int:
            return x * 3
            
        def sum_func(x: int, y: int) -> int:
            return x + y
            
        numbers = TypedVar('numbers', List[int])
        filtered = Var('filtered')
        tripled = Var('tripled')
        total = Var('total')
        
        result = run([
            eq(numbers, [1, 2, 3, 4, 5, 6]),
            apply(filter, [is_even, numbers], filtered),
            apply(map, [triple, filtered], tripled),
            apply(reduce, [sum_func, tripled], total)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['filtered'], [2, 4, 6])
        self.assertEqual(result[0]['tripled'], [6, 12, 18])
        self.assertEqual(result[0]['total'], 36)

class TestAdvancedPipelines(unittest.TestCase):
    def test_complex_data_transformation_pipeline(self):
        """Test a complex pipeline that transforms data through multiple stages with type checking"""
        # Input processing stage
        input_data = TypedVar('input_data', List[Dict[str, Union[int, str]]])
        stage1_result = Var('stage1_result')
        stage2_result = Var('stage2_result')
        final_result = Var('final_result')
        
        def extract_numbers(data: List[Dict[str, Union[int, str]]]) -> List[int]:
            return [d['value'] for d in data if isinstance(d.get('value'), int)]
            
        def filter_even(numbers: List[int]) -> List[int]:
            return [n for n in numbers if n % 2 == 0]
            
        def calculate_stats(numbers: List[int]) -> Dict[str, Union[int, float]]:
            if not numbers:
                return {'count': 0, 'sum': 0, 'avg': 0.0}
            return {
                'count': len(numbers),
                'sum': sum(numbers),
                'avg': sum(numbers) / len(numbers)
            }
        
        result = run([
            eq(input_data, [
                {'id': 'a', 'value': 1},
                {'id': 'b', 'value': 'skip'},
                {'id': 'c', 'value': 4},
                {'id': 'd', 'value': 6}
            ]),
            apply(extract_numbers, [input_data], stage1_result),
            apply(filter_even, [stage1_result], stage2_result),
            apply(calculate_stats, [stage2_result], final_result)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['stage1_result'], [1, 4, 6])
        self.assertEqual(result[0]['stage2_result'], [4, 6])
        self.assertEqual(result[0]['final_result'], {
            'count': 2,
            'sum': 10,
            'avg': 5.0
        })

    def test_branching_pipeline_with_type_constraints(self):
        """Test a pipeline that branches based on type constraints and merges results"""
        input_var = Var('input')
        branch1_result = TypedVar('branch1_result', List[int])
        branch2_result = TypedVar('branch2_result', List[str])
        merged_result = Var('merged_result')
        
        def process_numbers(data: List[Any]) -> List[int]:
            return [x * 2 for x in data if isinstance(x, int)]
            
        def process_strings(data: List[Any]) -> List[str]:
            return [s.upper() for s in data if isinstance(s, str)]
            
        def merge_results(nums: List[int], strs: List[str]) -> Dict[str, List[Any]]:
            return {
                'numbers': nums,
                'strings': strs
            }
        
        result = run([
            eq(input_var, [1, "hello", 2, "world", 3]),
            apply(process_numbers, [input_var], branch1_result),
            apply(process_strings, [input_var], branch2_result),
            apply(merge_results, [branch1_result, branch2_result], merged_result)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['branch1_result'], [2, 4, 6])
        self.assertEqual(result[0]['branch2_result'], ["HELLO", "WORLD"])
        self.assertEqual(result[0]['merged_result'], {
            'numbers': [2, 4, 6],
            'strings': ["HELLO", "WORLD"]
        })

    def test_recursive_pipeline_with_accumulation(self):
        """Test a pipeline that processes nested structures recursively"""
        input_tree = TypedVar('input_tree', Dict[str, Union[int, Dict]])
        processed_tree = Var('processed_tree')
        leaf_sum = Var('leaf_sum')
        
        def process_tree(tree: Dict) -> Dict:
            result = {}
            for k, v in tree.items():
                if isinstance(v, dict):
                    result[k] = process_tree(v)
                elif isinstance(v, int):
                    result[k] = v * 2
            return result
            
        def sum_leaves(tree: Dict) -> int:
            total = 0
            for v in tree.values():
                if isinstance(v, dict):
                    total += sum_leaves(v)
                elif isinstance(v, int):
                    total += v
            return total
        
        result = run([
            eq(input_tree, {
                'a': 1,
                'b': {
                    'c': 2,
                    'd': {
                        'e': 3
                    }
                }
            }),
            apply(process_tree, [input_tree], processed_tree),
            apply(sum_leaves, [processed_tree], leaf_sum)
        ])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['processed_tree'], {
            'a': 2,
            'b': {
                'c': 4,
                'd': {
                    'e': 6
                }
            }
        })
        self.assertEqual(result[0]['leaf_sum'], 12)

class TestAdvancedBranching(unittest.TestCase):
    def test_conditional_type_branching(self):
        """Test branching logic based on runtime type checking"""
        input_val = Var('input_val')
        type_var = Var('type_var')
        result = Var('result')
        
        def process_by_type(val: Any) -> Dict[str, Any]:
            if isinstance(val, int):
                return {'type': 'number', 'value': val * 2}
            elif isinstance(val, str):
                return {'type': 'string', 'value': val.upper()}
            elif isinstance(val, list):
                return {'type': 'list', 'value': len(val)}
            return {'type': 'unknown', 'value': None}
        
        # Test with different input types
        for test_input, expected_type, expected_value in [
            (42, 'number', 84),
            ("hello", 'string', "HELLO"),
            ([1, 2, 3], 'list', 3)
        ]:
            with self.subTest(input=test_input):
                r = run([
                    eq(input_val, test_input),
                    type_of(input_val, type_var),
                    apply(process_by_type, [input_val], result)
                ])
                
                self.assertEqual(len(r), 1)
                self.assertEqual(r[0]['result']['type'], expected_type)
                self.assertEqual(r[0]['result']['value'], expected_value)

    def test_dynamic_pipeline_construction(self):
        """Test dynamically constructing pipelines based on input types"""
        input_data = Var('input_data')
        pipeline_type = Var('pipeline_type')
        intermediate = Var('intermediate')
        result = Var('result')
        
        def determine_pipeline(data: Any) -> str:
            if all(isinstance(x, int) for x in data):
                return 'numeric'
            elif all(isinstance(x, str) for x in data):
                return 'text'
            return 'mixed'
            
        def numeric_pipeline(data: List[int]) -> Dict[str, float]:
            return {'mean': sum(data) / len(data), 'sum': sum(data)}
            
        def text_pipeline(data: List[str]) -> Dict[str, Union[List[str], int]]:
            return {'upper': [s.upper() for s in data], 'lengths': [len(s) for s in data]}
            
        def mixed_pipeline(data: List[Any]) -> Dict[str, List[Any]]:
            nums = [x for x in data if isinstance(x, int)]
            texts = [x for x in data if isinstance(x, str)]
            return {'numbers': nums, 'texts': texts}
        
        # Test numeric pipeline
        r1 = run([
            eq(input_data, [1, 2, 3, 4]),
            apply(determine_pipeline, [input_data], pipeline_type),
            eq(pipeline_type, 'numeric'),
            apply(numeric_pipeline, [input_data], result)
        ])
        
        self.assertEqual(r1[0]['result'], {'mean': 2.5, 'sum': 10})
        
        # Test text pipeline
        r2 = run([
            eq(input_data, ['a', 'b', 'c']),
            apply(determine_pipeline, [input_data], pipeline_type),
            eq(pipeline_type, 'text'),
            apply(text_pipeline, [input_data], result)
        ])
        
        self.assertEqual(r2[0]['result'], {
            'upper': ['A', 'B', 'C'],
            'lengths': [1, 1, 1]
        })
        
        # Test mixed pipeline
        r3 = run([
            eq(input_data, [1, 'a', 2, 'b']),
            apply(determine_pipeline, [input_data], pipeline_type),
            eq(pipeline_type, 'mixed'),
            apply(mixed_pipeline, [input_data], result)
        ])
        
        self.assertEqual(r3[0]['result'], {
            'numbers': [1, 2],
            'texts': ['a', 'b']
        })

if __name__ == '__main__':
    unittest.main() 