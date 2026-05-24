from typing import List, Tuple, Optional
from models import Move, SideType
from game_checkers import GameEngine
from constants import WHITE_CHECKERS, BLACK_CHECKERS, WHITE_REGULAR, BLACK_REGULAR, WHITE_QUEEN, BLACK_QUEEN


class CheckersAI:
    def __init__(self, engine: GameEngine, depth: int = 3):
        self.engine = engine
        self.depth = depth

    def get_best_move(self, side: SideType) -> Optional[List[Move]]:
        """Возвращает лучшую последовательность ходов для стороны."""
        _, best_seq = self._minimax(self.engine, self.depth, side, side,
                                    -float('inf'), float('inf'))
        return best_seq

    def _minimax(self, state: GameEngine, depth: int,
                 current_side: SideType, maximizing_side: SideType,
                 alpha: float, beta: float) -> Tuple[float, Optional[List[Move]]]:
        if depth == 0 or state.game_over:
            return self._evaluate(state, maximizing_side), None

        moves = state.get_all_full_moves(current_side)
        if not moves:
            score = -10000 if current_side == maximizing_side else 10000
            return score, None

        if current_side == maximizing_side:
            max_eval = -float('inf')
            best_move = None
            for seq in moves:
                new_state = self._apply_sequence(state, seq)
                eval_score, _ = self._minimax(new_state, depth - 1,
                                              new_state.current_player,
                                              maximizing_side, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = seq
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for seq in moves:
                new_state = self._apply_sequence(state, seq)
                eval_score, _ = self._minimax(new_state, depth - 1,
                                              new_state.current_player,
                                              maximizing_side, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = seq
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def _apply_sequence(self, state: GameEngine, sequence: List[Move]) -> GameEngine:
        new_state = state.copy()
        for move in sequence:
            new_state.apply_move(move)
        return new_state

    def _evaluate(self, state: GameEngine, side: SideType) -> float:
        """Оценочная функция: материал + лёгкие позиционные бонусы."""
        if state.game_over:
            return 10000 if state.winner == side else -10000

        score = 0.0
        # Материальный перевес
        if side == SideType.WHITE:
            score += state.field.white_score - state.field.black_score
            score += (state.field.captured_white_score - state.field.captured_black_score) * 0.5
        else:
            score += state.field.black_score - state.field.white_score
            score += (state.field.captured_black_score - state.field.captured_white_score) * 0.5

        # Позиционные бонусы (центр и продвижение)
        for y in range(state.field.height):
            for x in range(state.field.width):
                piece = state.field.type_at(x, y)
                if piece == 0:
                    continue
                # Центр (клетки 3-8 на доске 12x12)
                if 3 <= x <= 8 and 3 <= y <= 8:
                    if (side == SideType.WHITE and piece in WHITE_CHECKERS) or \
                       (side == SideType.BLACK and piece in BLACK_CHECKERS):
                        score += 0.1
                    else:
                        score -= 0.1
                # Продвижение вперёд
                if piece == WHITE_REGULAR and side == SideType.WHITE:
                    score += (state.field.height - y) * 0.05
                elif piece == BLACK_REGULAR and side == SideType.BLACK:
                    score += y * 0.05
        return score