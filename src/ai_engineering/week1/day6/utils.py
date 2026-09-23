def calculator(expression: str) -> float:
    try:
        return eval(expression)
    except:
        raise ValueError("Invalid expression")


def get_pricing(options: str) -> int:
    match options:
        case "iphone 17 pro":
            return 4000
        case "iphone 17":
            return 2000
        case "galaxy s26 ultra":
            return 5000
        case _:
            raise ValueError("Invalid option selected")
