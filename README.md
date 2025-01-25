# montyp

Montyp is a constraint-based unification engine for Python.

It is designed to be used in conjunction with the `typing` module to perform type inference and constraint satisfaction.

Name is after a **Mon**key with a **Typ**ewriter.

# Usage

## Quickstart
```python
import json
from montyp.engine import run, eq, type_of
from montyp.schemas import Var, TypedVar

# Define functions
def add(x: int, y: int) -> int:
    return x + y
    
# Define variables
x = TypedVar('x', int)
y = Var('y')
z = Var('z')
x_type = Var('x_type')
add_type = Var('add_type')

# Define constraints
result = run([
    eq(x, 42),
    eq(y, "Hello"),
    eq(z, (x, y)),
    type_of(x, x_type),
    type_of(add, add_type)
])
    
print("Demo result:\n", json.dumps(result, indent=2))
```

this should output:

```
Demo result:
 [
  {
    "y": "Hello",
    "x_type": "int",
    "add_type": "(int, int) -> int",
    "z": [
      42,
      "Hello"
    ],
    "x": 42
  }
]
```

Alternatively, you can run main.py to see the same result.

# Tests

To run the tests, run:

```
python -m tests.test_engine
```
