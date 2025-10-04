def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value between a lower and upper bound."""
    return max(lower, min(value, upper))
