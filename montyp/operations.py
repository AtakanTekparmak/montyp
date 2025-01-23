from typing import Any, Callable, List, Optional, Tuple, Union
from inspect import signature
from operator import add, mul
from math import sqrt

def string_reverse(s: str) -> str:
    return s[::-1]

class Function:
    """Wrapper for functions with additional properties for logical programming."""
    def __init__(self, fn: Callable, 
                 inverse: Optional[Union[Callable, List[Tuple[int, Callable]]]] = None,
                 domain: Optional[Callable[[List[Any]], bool]] = None):
        self.fn = fn
        self.inverse = inverse  # Either a single inverse function or list of (arg_index, inverse_fn)
        self.domain = domain or (lambda args: True)  # Domain constraint
        self.sig = signature(fn)
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

# Common arithmetic operations
ADD = Function(add, 
              inverse=[(0, lambda res, y: res - y),   # x + y = res -> x = res - y
                      (1, lambda res, x: res - x)])   # x + y = res -> y = res - x

MUL = Function(mul,
              inverse=[(0, lambda res, y: res / y if y != 0 else None),  # x * y = res -> x = res / y
                      (1, lambda res, x: res / x if x != 0 else None)],  # x * y = res -> y = res / x
              domain=lambda args: 0 not in args[1:])  # Prevent division by zero

# String operations
REVERSE = Function(string_reverse,
                  inverse=[(0, string_reverse)])  # reverse(x) = y -> x = reverse(y)

# Mathematical operations
SQUARE = Function(lambda x: x * x,
                 inverse=[(0, lambda res: sqrt(res) if res >= 0 else None)],
                 domain=lambda args: True) 