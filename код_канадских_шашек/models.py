from enum import Enum, auto

class SideType(Enum):
    WHITE = auto()
    BLACK = auto()

    def opposite(self):
        return SideType.BLACK if self == SideType.WHITE else SideType.WHITE

class CheckerType(Enum):
    NONE = 0
    WHITE_REGULAR = 1
    BLACK_REGULAR = 2
    WHITE_QUEEN = 3
    BLACK_QUEEN = 4

class Point:
    __slots__ = ('x', 'y')
    def __init__(self, x: int = -1, y: int = -1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Move:
    __slots__ = ('from_x', 'from_y', 'to_x', 'to_y')
    def __init__(self, from_x: int = -1, from_y: int = -1, to_x: int = -1, to_y: int = -1):
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y

    def __eq__(self, other):
        if not isinstance(other, Move):
            return NotImplemented
        return (self.from_x == other.from_x and self.from_y == other.from_y and
                self.to_x == other.to_x and self.to_y == other.to_y)

    def __hash__(self):
        return hash((self.from_x, self.from_y, self.to_x, self.to_y))

    def __repr__(self):
        return f"Move({self.from_x},{self.from_y} -> {self.to_x},{self.to_y})"

class Checker:
    __slots__ = ('type',)
    def __init__(self, type_: CheckerType = CheckerType.NONE):
        self.type = type_

def with_type(new_type: CheckerType) -> 'Checker':
    """Возвращает новый объект шашки с изменённым типом (для функционального стиля)."""
    return Checker(new_type)