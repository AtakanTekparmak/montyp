from montyp import run, eq, conde, Var, typeo

if __name__ == "__main__":
    x, y = Var(), Var()
    
    goals = [
        conde(
            [eq(x, 5)],  # First branch
            [eq(x, 6)]   # Second branch
        ),
        eq(y, [x, 6]),
        typeo(y, 'list')
    ]
    
    print("Solutions:")
    for sol in run(goals):
        x_val = sol.get(x, '?')
        y_val = sol.get(y, '?')
        print(f"x: {x_val}, y: {y_val}")