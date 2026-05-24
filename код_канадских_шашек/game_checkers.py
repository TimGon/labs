from typing import List, Optional, Tuple, Dict, Any

from constants import (
    CELL_SIZE, ANIMATION_SPEED, ANIMATION_DELAY,
    BORDER_WIDTH, MOVE_OFFSETS, WHITE_CHECKERS,
    BLACK_CHECKERS, WHITE_REGULAR, WHITE_QUEEN,
    BLACK_REGULAR, BLACK_QUEEN,SELECT_BORDER_COLOR,
    HOVER_BORDER_COLOR, POSSIBLE_MOVE_CIRCLE_COLOR,
    FIELD_COLORS
)
from models import SideType, CheckerType, Point, Move
from field import Field
from PIL import Image, ImageTk
from pathlib import Path
from time import sleep

import tkinter as tk
from tkinter import messagebox

class GameEngine:
    """Чистая игровая логика без привязки к GUI."""
    def __init__(self):
        self.field = Field()
        self.current_player = SideType.WHITE
        self._winner: Optional[SideType] = None
        self._game_over = False
    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def winner(self) -> Optional[SideType]:
        return self._winner

    def copy(self) -> 'GameEngine':
        """Создаёт копию состояния игры."""
        new_engine = GameEngine.__new__(GameEngine)
        new_engine.field = self.field.copy()
        new_engine.current_player = self.current_player
        new_engine._game_over = self._game_over
        new_engine._winner = self._winner
        return new_engine

    def get_possible_moves(self, side: Optional[SideType] = None) -> List[Move]:
        """
        Возвращает все возможные ходы для указанной стороны.
        Если есть обязательные взятия, возвращает только их.
        """
        if side is None:
            side = self.current_player
        capturing = self._get_capturing_moves(side)
        if capturing:
            return capturing
        return self._get_non_capturing_moves(side)

    def _get_capturing_moves(self, side: SideType) -> List[Move]:
        moves = []
        friendly = WHITE_CHECKERS if side == SideType.WHITE else BLACK_CHECKERS
        enemy = BLACK_CHECKERS if side == SideType.WHITE else WHITE_CHECKERS

        for y in range(self.field.height):
            for x in range(self.field.width):
                piece = self.field.type_at(x, y)
                if piece not in friendly:
                    continue
                if piece in (WHITE_REGULAR, BLACK_REGULAR):
                    moves.extend(self._man_captures(x, y, enemy))
                else:
                    moves.extend(self._king_captures(x, y, friendly, enemy))
        return moves

    def _get_non_capturing_moves(self, side: SideType) -> List[Move]:
        moves = []
        friendly = WHITE_CHECKERS if side == SideType.WHITE else BLACK_CHECKERS

        for y in range(self.field.height):
            for x in range(self.field.width):
                piece = self.field.type_at(x, y)
                if piece not in friendly:
                    continue
                if piece in (WHITE_REGULAR, BLACK_REGULAR):
                    moves.extend(self._man_simple_moves(x, y, side))
                else:
                    moves.extend(self._king_simple_moves(x, y))
        return moves

    def _man_captures(self, x: int, y: int, enemy_set: frozenset) -> List[Move]:
        moves = []
        for offset in MOVE_OFFSETS:
            nx, ny = x + offset.x, y + offset.y
            jx, jy = x + offset.x * 2, y + offset.y * 2
            if not self.field.is_within(jx, jy):
                continue
            if self.field.type_at(nx, ny) in enemy_set and self.field.type_at(jx, jy) == CheckerType.NONE:
                moves.append(Move(x, y, jx, jy))
        return moves

    def _king_captures(self, x: int, y: int, friendly_set: frozenset, enemy_set: frozenset) -> List[Move]:
        moves = []
        for offset in MOVE_OFFSETS:
            # Ищем первую вражескую шашку за которой следует пустая клетка
            step = 1
            while True:
                cx, cy = x + offset.x * step, y + offset.y * step
                if not self.field.is_within(cx, cy):
                    break
                cell = self.field.type_at(cx, cy)
                if cell in friendly_set:
                    break
                if cell in enemy_set:
                    # Нашли врага, ищем пустую клетку дальше
                    j_step = step + 1
                    while True:
                        jx, jy = x + offset.x * j_step, y + offset.y * j_step
                        if not self.field.is_within(jx, jy):
                            break
                        if self.field.type_at(jx, jy) == CheckerType.NONE:
                            moves.append(Move(x, y, jx, jy))
                        else:
                            break  # занято – дальше не прыгнуть
                        j_step += 1
                    break
                step += 1
        return moves

    def _man_simple_moves(self, x: int, y: int, side: SideType) -> List[Move]:
        moves = []
        # Обычная шашка ходит только вперёд (для белых y-1, для чёрных y+1)
        dirs = MOVE_OFFSETS[:2] if side == SideType.WHITE else MOVE_OFFSETS[2:]
        for offset in dirs:
            nx, ny = x + offset.x, y + offset.y
            if self.field.is_within(nx, ny) and self.field.type_at(nx, ny) == CheckerType.NONE:
                moves.append(Move(x, y, nx, ny))
        return moves

    def _king_simple_moves(self, x: int, y: int) -> List[Move]:
        moves = []
        for offset in MOVE_OFFSETS:
            step = 1
            while True:
                nx, ny = x + offset.x * step, y + offset.y * step
                if not self.field.is_within(nx, ny):
                    break
                if self.field.type_at(nx, ny) == CheckerType.NONE:
                    moves.append(Move(x, y, nx, ny))
                else:
                    break
                step += 1
        return moves

    def apply_move(self, move: Move) -> Tuple[bool, Optional[CheckerType]]:
        """
        Применяет ход, обновляет состояние.
        Возвращает (было_ли_взятие, тип_съеденной_шашки).
        """
        killed = self.field.move(move.from_x, move.from_y, move.to_x, move.to_y,
                                 capturing_side=self.current_player)
        has_killed = killed is not None

        # Проверяем, может ли шашка продолжить взятие с новой клетки
        can_continue = has_killed and self._has_further_captures(move.to_x, move.to_y)

        # Превращаем в дамку только если серия взятий завершена
        if not can_continue:
            self.field.set_queen(move.to_x, move.to_y)
        # Проверка окончания игры
        self._check_game_over()

        # Переключение игрока, если нет продолжения взятия той же шашкой
        if not has_killed or not self._has_further_captures(move.to_x, move.to_y):
            self.current_player = self.current_player.opposite()
            self._check_game_over()  # для нового игрока

        return has_killed, killed

    def _has_further_captures(self, x: int, y: int) -> bool:
        """Проверяет, может ли шашка на (x, y) продолжить взятие."""
        piece = self.field.type_at(x, y)
        if piece == CheckerType.NONE:
            return False
        friendly = WHITE_CHECKERS if piece in WHITE_CHECKERS else BLACK_CHECKERS
        enemy = BLACK_CHECKERS if piece in WHITE_CHECKERS else WHITE_CHECKERS
        if piece in (WHITE_REGULAR, BLACK_REGULAR):
            return bool(self._man_captures(x, y, enemy))
        else:
            return bool(self._king_captures(x, y, friendly, enemy))

    def _check_game_over(self) -> None:
        """Проверяет, остались ли ходы у текущего игрока."""
        moves = self.get_possible_moves()
        if not moves:
            self._game_over = True
            self._winner = SideType.WHITE if self.current_player == SideType.BLACK else SideType.BLACK

    # ----- Методы для Reinforcement Learning -----
    def get_state(self) -> Dict[str, Any]:
        """Возвращает состояние игры в виде словаря."""
        return {
            'board': [row[:] for row in self.field._board],
            'current_player': self.current_player.value,
            'white_score': self.field.white_score,
            'black_score': self.field.black_score,
        }

    def get_valid_actions(self) -> List[Move]:
        """Список допустимых действий (ходов) для текущего игрока."""
        return self.get_possible_moves()

    def step(self, action: Move) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Выполняет ход, возвращает (новое_состояние, награда, done, info).
        """
        has_killed, killed_type = self.apply_move(action)
        reward = 0.0
        if killed_type is not None:
            # Награда за взятие
            reward = 3.0 if killed_type in (WHITE_QUEEN, BLACK_QUEEN) else 1.0
        if self._game_over:
            # Победа / поражение
            if self._winner == SideType.WHITE:
                reward += 100.0 if self.current_player == SideType.WHITE else -100.0
            else:
                reward += 100.0 if self.current_player == SideType.BLACK else -100.0

        return self.get_state(), reward, self._game_over, {'killed': killed_type is not None}

    def reset(self) -> Dict[str, Any]:
        """Сбрасывает игру в начальное состояние."""
        self.__init__()
        return self.get_state()

    def get_all_full_moves(self, side: SideType) -> List[List[Move]]:
        """Возвращает все возможные полные ходы (последовательности Move, образующие один игровой ход)."""
        full_moves = []
        for y in range(self.field.height):
            for x in range(self.field.width):
                piece = self.field.type_at(x, y)
                if (side == SideType.WHITE and piece in WHITE_CHECKERS) or \
                        (side == SideType.BLACK and piece in BLACK_CHECKERS):
                    sequences = self._generate_sequences_from_piece(x, y, side)
                    full_moves.extend(sequences)
        if not full_moves:
            return []
        # Правило максимального взятия: оставляем только ходы с наибольшим количеством взятий
        max_captures = max(len(seq) for seq in full_moves)
        return [seq for seq in full_moves if len(seq) == max_captures]

    def _generate_sequences_from_piece(self, x: int, y: int, side: SideType) -> List[List[Move]]:
        sequences = []
        capturing_moves = self._get_capturing_moves_from_piece(x, y, side)
        if not capturing_moves:
            simple_moves = self._get_simple_moves_from_piece(x, y, side)
            for move in simple_moves:
                sequences.append([move])
            return sequences
        for move in capturing_moves:
            # Применяем ход на копии, чтобы проверить продолжение
            new_state = self.copy()
            new_state.apply_move(move)
            if new_state._has_further_captures(move.to_x, move.to_y):
                sub_sequences = new_state._generate_sequences_from_piece(move.to_x, move.to_y, side)
                for sub_seq in sub_sequences:
                    sequences.append([move] + sub_seq)
            else:
                sequences.append([move])
        return sequences

    def _get_capturing_moves_from_piece(self, x: int, y: int, side: SideType) -> List[Move]:
        enemy = BLACK_CHECKERS if side == SideType.WHITE else WHITE_CHECKERS
        piece = self.field.type_at(x, y)
        if piece in (WHITE_REGULAR, BLACK_REGULAR):
            return self._man_captures(x, y, enemy)
        else:
            friendly = WHITE_CHECKERS if side == SideType.WHITE else BLACK_CHECKERS
            return self._king_captures(x, y, friendly, enemy)

    def _get_simple_moves_from_piece(self, x: int, y: int, side: SideType) -> List[Move]:
        piece = self.field.type_at(x, y)
        if piece in (WHITE_REGULAR, BLACK_REGULAR):
            return self._man_simple_moves(x, y, side)
        else:
            return self._king_simple_moves(x, y)


# ----- GUI часть (только визуализация) -----
class Game(GameEngine):
    """Расширяет GameEngine визуализацией на Canvas."""
    def __init__(self, canvas: tk.Canvas):
        super().__init__()
        self.canvas = canvas
        self.hovered_cell = Point()
        self.selected_cell = Point()
        self.animated_cell = Point()
        self.is_animating = False
        self._images = {}
        self.cell_size = CELL_SIZE
        self._init_images()

    def _init_images(self):
        size = self.cell_size
        assets = Path('assets')
        self._images = {
            CheckerType.WHITE_REGULAR: ImageTk.PhotoImage(
                Image.open(assets / 'white-regular.png').resize((size, size), Image.Resampling.LANCZOS)),
            CheckerType.BLACK_REGULAR: ImageTk.PhotoImage(
                Image.open(assets / 'black-regular.png').resize((size, size), Image.Resampling.LANCZOS)),
            CheckerType.WHITE_QUEEN: ImageTk.PhotoImage(
                Image.open(assets / 'white-queen.png').resize((size, size), Image.Resampling.LANCZOS)),
            CheckerType.BLACK_QUEEN: ImageTk.PhotoImage(
                Image.open(assets / 'black-queen.png').resize((size, size), Image.Resampling.LANCZOS)),
        }

    def draw(self):
        """Отрисовка поля и шашек."""
        self.canvas.delete('all')
        self._draw_grid()
        self._draw_checkers()
        if self.selected_cell.x != -1:
            self._draw_possible_moves()

    def _draw_grid(self):
        cell = self.cell_size
        for y in range(self.field.height):
            for x in range(self.field.width):
                color = FIELD_COLORS[(y + x) % 2]
                self.canvas.create_rectangle(
                    x * cell, y * cell,
                    x * cell + cell, y * cell + cell,
                    fill=color, width=0, tags='board'
                )
                # Рамка при наведении/выборе
                if x == self.selected_cell.x and y == self.selected_cell.y:
                    self.canvas.create_rectangle(
                        x * cell + BORDER_WIDTH//2, y * cell + BORDER_WIDTH//2,
                        x * cell + cell - BORDER_WIDTH//2, y * cell + cell - BORDER_WIDTH//2,
                        outline=SELECT_BORDER_COLOR, width=BORDER_WIDTH, tags='border'
                    )
                elif x == self.hovered_cell.x and y == self.hovered_cell.y:
                    self.canvas.create_rectangle(
                        x * cell + BORDER_WIDTH//2, y * cell + BORDER_WIDTH//2,
                        x * cell + cell - BORDER_WIDTH//2, y * cell + cell - BORDER_WIDTH//2,
                        outline=HOVER_BORDER_COLOR, width=BORDER_WIDTH, tags='border'
                    )

    def _draw_checkers(self):
        cell = self.cell_size
        for y in range(self.field.height):
            for x in range(self.field.width):
                if x == self.animated_cell.x and y == self.animated_cell.y:
                    continue
                piece = self.field.type_at(x, y)
                if piece != CheckerType.NONE:
                    self.canvas.create_image(
                        x * cell, y * cell,
                        image=self._images[piece], anchor='nw', tags='checker'
                    )

    def _draw_possible_moves(self):
        moves = self.get_possible_moves()
        cell = self.cell_size
        for move in moves:
            if move.from_x == self.selected_cell.x and move.from_y == self.selected_cell.y:
                self.canvas.create_oval(
                    move.to_x * cell + cell/3, move.to_y * cell + cell/3,
                    move.to_x * cell + cell - cell/3, move.to_y * cell + cell - cell/3,
                    fill=POSSIBLE_MOVE_CIRCLE_COLOR, width=0, tags='possible'
                )

    def animate_move(self, move: Move):
        self.is_animating = True
        self.animated_cell = Point(move.from_x, move.from_y)
        self.draw()
        cell = self.cell_size
        img = self._images[self.field.type_at(move.from_x, move.from_y)]
        anim = self.canvas.create_image(
            move.from_x * cell, move.from_y * cell,
            image=img, anchor='nw', tags='animated'
        )
        dx = 1 if move.to_x > move.from_x else -1
        dy = 1 if move.to_y > move.from_y else -1
        steps = abs(move.to_x - move.from_x)
        for _ in range(steps):
            for _ in range(int(100 / ANIMATION_SPEED)):
                self.canvas.move(anim, ANIMATION_SPEED/100 * cell * dx, ANIMATION_SPEED/100 * cell * dy)
                self.canvas.update()
                sleep(ANIMATION_DELAY)
        self.animated_cell = Point()
        self.is_animating = False

    def handle_player_turn(self, move: Move):
        """Обработка хода игрока с анимацией."""
        if self.is_animating:
            return
        self.animate_move(move)
        has_killed, _ = self.apply_move(move)
        self.selected_cell = Point()
        if has_killed and self._has_further_captures(move.to_x, move.to_y):
            # Продолжение взятия той же шашкой
            self.selected_cell = Point(move.to_x, move.to_y)
        self.draw()
        if self.game_over:
            self._show_game_over()

    def _show_game_over(self):
        winner = "Белые" if self.winner == SideType.WHITE else "Чёрные"
        tk.messagebox.showinfo("Конец игры", f"{winner} выиграли!")
        # Не вызываем __init__, просто сбрасываем состояние через reset()
        self.reset()
        self.draw()

    # Обработчики событий мыши
    def mouse_move(self, event: tk.Event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if (x, y) != (self.hovered_cell.x, self.hovered_cell.y):
            self.hovered_cell = Point(x, y)
            self.draw()

    def mouse_down(self, event: tk.Event):
        if self.is_animating or self.game_over:
            return
        x, y = event.x // self.cell_size, event.y // self.cell_size
        if not self.field.is_within(x, y):
            return

        moves = self.get_possible_moves()
        # Если есть выбранная клетка и ход из неё
        if self.selected_cell.x != -1:
            move = Move(self.selected_cell.x, self.selected_cell.y, x, y)
            if move in moves:
                self.handle_player_turn(move)
                return

        # Выбор новой шашки
        piece = self.field.type_at(x, y)
        friendly = WHITE_CHECKERS if self.current_player == SideType.WHITE else BLACK_CHECKERS
        if piece in friendly:
            # Проверяем, есть ли ходы для этой шашки
            for m in moves:
                if m.from_x == x and m.from_y == y:
                    self.selected_cell = Point(x, y)
                    self.draw()
                    return
        self.selected_cell = Point()
        self.draw()

    def make_move(self, move: Move):
        """Выполнить один ход с анимацией и обновить состояние."""
        if self.is_animating:
            return
        self.animate_move(move)
        has_killed, _ = self.apply_move(move)
        self.selected_cell = Point()
        if has_killed and self._has_further_captures(move.to_x, move.to_y):
            self.selected_cell = Point(move.to_x, move.to_y)
        self.draw()
        if self.game_over:
            self._show_game_over()

    def perform_full_move(self, sequence: List[Move]):
        """Выполнить последовательность ходов (полный ход) с анимацией."""
        for move in sequence:
            self.make_move(move)