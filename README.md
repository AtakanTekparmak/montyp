# montyp

Montyp is a constraint-based unification engine for Python.

It is designed to be used in conjunction with the `typing` module to perform type inference and constraint satisfaction.

Name is after a **Mon**key with a **Typ**ewriter.

# Usage

## Quickstart
```python
import json
from montyp.engine import run, eq, type_of, apply
from montyp.schemas import Var, TypedVar

# Define functions
def add(x: int, y: int) -> int:
    return x + y
    
# Define variables
x = TypedVar('x', int)
y = Var('y')  # Untyped variable - will be deduced
z = Var('z')

x_type = Var('x_type')
add_type = Var('add_type')
y_type = Var('y_type')

# Define constraints
result = run([
    eq(x, 42),
    eq(z, (x, y)),
    type_of(x, x_type),
    type_of(add, add_type),    # We know we want k to be 52
    apply(add, [x, y], 52),  # This will deduce y = 10 since 42 + 10 = 52
    type_of(y, y_type)
])
    
print(json.dumps(result, indent=2))
```

this should output:

```json
[
  {
    "z": [
      42,
      10
    ],
    "add_type": "(int, int) -> int",
    "x_type": "int",
    "x": 42,
    "y_type": "int",
    "y": 10
  }
]
```

The example demonstrates:
1. Type constraints with `TypedVar`
2. Type inference with `type_of`
3. Function type inference
4. Variable deduction through function application
5. Tuple construction with variables

In this example, the engine:
- Knows x is 42 (from explicit constraint)
- Knows k should be 52 (from explicit constraint)
- Deduces that y must be 10 (since 42 + 10 = 52)
- Creates tuple z = (42, 10)
- Infers types for x and the add function

Alternatively, you can run main.py to see the same result.

# Tests

To run the tests, run:

```bash
python -m tests.test_engine
```
