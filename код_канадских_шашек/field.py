from typing import List, Optional
from models import CheckerType, SideType
from constants import BOARD_WIDTH, BOARD_HEIGHT, WHITE_CHECKERS, BLACK_CHECKERS, WHITE_QUEEN, BLACK_QUEEN


class Field:
    __slots__ = ('_board', '_white_count', '_black_count',
                 '_white_score', '_black_score',
                 '_captured_white_score', '_captured_black_score')

    def __init__(self):

        self._board: List[List[CheckerType]] = []
        self._white_count = 0
        self._black_count = 0
        self._white_score = 0          # ценность своих оставшихся фигур (для информации)
        self._black_score = 0
        self._captured_white_score = 0 # очки, заработанные белыми за взятие чёрных
        self._captured_black_score = 0
        self._generate()

    @property
    def width(self) -> int:
        return BOARD_WIDTH

    @property
    def height(self) -> int:
        return BOARD_HEIGHT

    @property
    def white_count(self) -> int:
        return self._white_count

    @property
    def black_count(self) -> int:
        return self._black_count

    @property
    def white_score(self) -> int:
        return self._white_score

    @property
    def black_score(self) -> int:
        return self._black_score

    @property
    def captured_white_score(self) -> int:
        return self._captured_white_score

    @property
    def captured_black_score(self) -> int:
        return self._captured_black_score

    def _add_capture(self, killed_type: CheckerType, capturing_side: SideType) -> None:
        """Начисляет очки за взятие."""
        points = 3 if killed_type in (WHITE_QUEEN, BLACK_QUEEN) else 1
        if capturing_side == SideType.WHITE:
            self._captured_white_score += points
        else:
            self._captured_black_score += points

    def _generate(self) -> None:
        """Начальная расстановка шашек."""
        self._board = [[CheckerType.NONE for _ in range(self.width)] for _ in range(self.height)]
        self._white_count = 0
        self._black_count = 0
        self._white_score = 0
        self._black_score = 0
        self._captured_white_score = 0
        self._captured_black_score = 0

        for y in range(self.height):
            for x in range(self.width):
                if (y + x) % 2 == 1:
                    if y < 5:
                        self._set_type(x, y, CheckerType.BLACK_REGULAR)
                    elif y >= self.height - 5:
                        self._set_type(x, y, CheckerType.WHITE_REGULAR)

    def _set_type(self, x: int, y: int, new_type: CheckerType) -> None:
        """Устанавливает тип шашки и обновляет счётчики."""
        old_type = self._board[y][x]

        # Уменьшаем счётчики для старого типа
        if old_type in WHITE_CHECKERS:
            self._white_count -= 1
            self._white_score -= 3 if old_type == CheckerType.WHITE_QUEEN else 1
        elif old_type in BLACK_CHECKERS:
            self._black_count -= 1
            self._black_score -= 3 if old_type == CheckerType.BLACK_QUEEN else 1

        # Увеличиваем счётчики для нового типа
        if new_type in WHITE_CHECKERS:
            self._white_count += 1
            self._white_score += 3 if new_type == CheckerType.WHITE_QUEEN else 1
        elif new_type in BLACK_CHECKERS:
            self._black_count += 1
            self._black_score += 3 if new_type == CheckerType.BLACK_QUEEN else 1

        self._board[y][x] = new_type

    def type_at(self, x: int, y: int) -> CheckerType:
        return self._board[y][x]

    def is_within(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def move(self, from_x: int, from_y: int, to_x: int, to_y: int,
             capturing_side: Optional[SideType] = None) -> Optional[CheckerType]:
        """
        Перемещает шашку. Возвращает тип съеденной шашки (если было взятие) или None.
        Параметр capturing_side нужен для начисления очков.
        """
        if not self.is_within(from_x, from_y) or not self.is_within(to_x, to_y):
            return None

        moving_type = self.type_at(from_x, from_y)
        if moving_type == CheckerType.NONE:
            return None

        # Удаляем шашку с начальной клетки
        self._set_type(from_x, from_y, CheckerType.NONE)

        killed_type = None
        dx = 1 if to_x > from_x else -1
        dy = 1 if to_y > from_y else -1
        steps = abs(to_x - from_x)

        if steps > 1:
            for step in range(1, steps):
                cx = from_x + dx * step
                cy = from_y + dy * step
                if not self.is_within(cx, cy):
                    continue
                cell_type = self.type_at(cx, cy)
                if cell_type != CheckerType.NONE:
                    killed_type = cell_type
                    self._set_type(cx, cy, CheckerType.NONE)
                    if capturing_side is not None:
                        self._add_capture(killed_type, capturing_side)
                    break

        self._set_type(to_x, to_y, moving_type)
        return killed_type

    def set_queen(self, x: int, y: int) -> None:
        """Превращает шашку в дамку, если она на последней горизонтали."""
        t = self.type_at(x, y)
        if t == CheckerType.WHITE_REGULAR and y == 0:
            self._set_type(x, y, CheckerType.WHITE_QUEEN)
        elif t == CheckerType.BLACK_REGULAR and y == self.height - 1:
            self._set_type(x, y, CheckerType.BLACK_QUEEN)

    def copy(self) -> 'Field':
        """Создаёт глубокую копию поля."""
        new_field = Field.__new__(Field)
        new_field._board = [row[:] for row in self._board]
        new_field._white_count = self._white_count
        new_field._black_count = self._black_count
        new_field._white_score = self._white_score
        new_field._black_score = self._black_score
        new_field._captured_white_score = self._captured_white_score
        new_field._captured_black_score = self._captured_black_score
        return new_field

    def __repr__(self):
        return f"<Field w={self.white_count} b={self.black_count}>"