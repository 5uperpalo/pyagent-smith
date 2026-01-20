from langchain.tools import tool


def _evaluate_expression(expression: str) -> str:
    import ast
    import operator as op

    allowed_ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
        ast.FloorDiv: op.floordiv,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants are allowed")
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError("Operator not allowed")
            return allowed_ops[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError("Unary operator not allowed")
            return allowed_ops[op_type](operand)
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree))


@tool("calculator", return_direct=False)
def calculator(expression: str, verbose: bool = False) -> str:
    """Evaluate a basic arithmetic expression (e.g., "(2 + 3) * 7 - 4/2")."""
    try:
        result = _evaluate_expression(expression)
        if verbose:
            print(f"calculator result: {result}")
        return result
    except Exception as e:
        return f"Calculator error: {e}"
