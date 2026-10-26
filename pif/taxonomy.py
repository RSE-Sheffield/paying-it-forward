"""The label scheme: nine categories from a software axis crossed with a data axis.

The scheme is fixed across criteria versions -- a criteria version changes what
each level means, not which levels exist -- so this is the one place the codes
are defined.
"""

SOFTWARE = {"S1": "Software generated", "S2": "Software used", "S3": "No software"}
DATA = {"D1": "Data generated", "D2": "Data used", "D3": "No data"}

# Data-major order (S1D1, S2D1, S3D1, S1D2, ...), as used since spike 0. Ties
# between categories are always broken in this order.
CATEGORIES = [s + d for d in DATA for s in SOFTWARE]
VALID = frozenset(CATEGORIES)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "reasoning": {"type": "string"},
    },
    "required": ["category", "reasoning"],
}


def software(category: str) -> str:
    return category[:2]


def data(category: str) -> str:
    return category[2:]


def label(category: str) -> str:
    """S1D2 -> 'Software generated, data used'."""
    return f"{SOFTWARE[software(category)]}, {DATA[data(category)].lower()}"
