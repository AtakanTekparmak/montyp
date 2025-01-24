from typing import Any, Callable, List, Optional, Tuple, Union, TypeVar, Protocol
from inspect import signature
from operator import add, mul, sub, truediv
from math import sqrt

T = TypeVar('T')
U = TypeVar('U')

class InverseFn(Protocol):
    """Protocol for inverse functions that can fail by returning None."""
    def __call__(self, result: Any, *args: Any) -> Optional[Any]: ...

def string_reverse(s: str) -> str:
    """Reverse a string.
    
    Args:
        s: The input string to reverse
    
    Returns:
        The reversed string
    """
    return s[::-1]

class Function:
    """Wrapper for functions with additional properties for logical programming.
    
    This class extends regular Python functions with:
        - Inverse functions for backward computation
        - Domain constraints for input validation
    """
    
    def __init__(self, 
                 fn: Callable[..., Any],
                 inverse: Optional[Union[Callable[..., Any], List[Tuple[int, InverseFn]]]] = None,
                 domain: Optional[Callable[[List[Any]], bool]] = None) -> None:
        """Initialize a logical function.
        
        Args:
            fn: The forward function to wrap
            inverse: Either a single inverse function or a list of (argument_index, inverse_function) pairs
            domain: Optional function that takes arguments and returns whether they're valid
        """
        self.fn = fn
        self.inverse = inverse  # Either a single inverse function or list of (arg_index, inverse_fn)
        self.domain = domain or (lambda args: True)  # Domain constraint
        self.sig = signature(fn)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments."""
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
                 inverse=[(0, lambda res: sqrt(res) if res >= 0 else None)],  # x² = res -> x = ±√res
                 domain=lambda args: True) 

# Add these new function definitions at the bottom of the file
SUB = Function(sub,
              inverse=[
                  (0, lambda res, y: res + y),   # x - y = res → x = res + y
                  (1, lambda res, x: x - res)    # x - y = res → y = x - res
              ],
              domain=lambda args: True)

DIV = Function(truediv,
              inverse=[
                  (0, lambda res, y: res * y if y != 0 else None),  # x / y = res → x = res * y
                  (1, lambda res, x: x / res if res != 0 else None)  # x / y = res → y = x / res
              ],
              domain=lambda args: args[1] != 0)  # Prevent division by zero 