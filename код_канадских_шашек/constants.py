from models import CheckerType, Point

# Размеры поля
BOARD_WIDTH = 12
BOARD_HEIGHT = 12
CELL_SIZE = 75          # базовая, может масштабироваться

# Анимация
ANIMATION_SPEED = 4
ANIMATION_DELAY = 0.01

# Визуальные настройки
BORDER_WIDTH = 4
FIELD_COLORS = ['#E7CFA9', '#927456']
HOVER_BORDER_COLOR = '#54b346'
SELECT_BORDER_COLOR = '#944444'
POSSIBLE_MOVE_CIRCLE_COLOR = '#944444'

# Направления движения (диагонали)
MOVE_OFFSETS = [
    Point(-1, -1),
    Point(1, -1),
    Point(-1, 1),
    Point(1, 1)
]

# Множества типов шашек для быстрой проверки
WHITE_REGULAR = CheckerType.WHITE_REGULAR
WHITE_QUEEN  = CheckerType.WHITE_QUEEN
BLACK_REGULAR = CheckerType.BLACK_REGULAR
BLACK_QUEEN  = CheckerType.BLACK_QUEEN

WHITE_CHECKERS = frozenset({WHITE_REGULAR, WHITE_QUEEN})
BLACK_CHECKERS = frozenset({BLACK_REGULAR, BLACK_QUEEN})
ALL_CHECKERS = WHITE_CHECKERS | BLACK_CHECKERS