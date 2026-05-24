import time
from game_checkers import GameEngine
from ai import CheckersAI
from dqn_agent import DQNAgent
from models import SideType


def run_match_minmax_vs_dqn(num_games=10, callback=print):
    """
    Проводит матч между MinMax (depth=3) и DQN (обученная модель).
    Возвращает словарь со статистикой и печатает подробные метрики.
    """
    # Загружаем DQN агента
    dqn = DQNAgent()
    try:
        dqn.load_model("dqn_model.pth")
        dqn.epsilon = 0.0
    except:
        callback("Внимание: не удалось загрузить dqn_model.pth, используется неподготовленная модель.")

    # Инициализация статистики
    stats = {
        "minmax_wins": 0,
        "dqn_wins": 0,
        "draws": 0,
        "minmax_total_score": 0,
        "dqn_total_score": 0,
        "total_moves": 0,
        "minmax_move_times": [],  # время каждого хода MinMax
        "dqn_move_times": []  # время каждого хода DQN
    }

    # Для каждой партии
    for game_idx in range(num_games):
        # Чередуем, кто начинает (и, соответственно, за кого играет MinMax)
        # В нечётных партиях MinMax играет белыми, в чётных – чёрными
        minmax_color = SideType.WHITE if game_idx % 2 == 0 else SideType.BLACK
        dqn_color = minmax_color.opposite()

        callback(f"\n--- Партия {game_idx + 1} ---")
        callback(f"MinMax играет за {'Белых' if minmax_color == SideType.WHITE else 'Чёрных'}")
        callback(f"DQN играет за {'Белых' if dqn_color == SideType.WHITE else 'Чёрных'}")

        env = GameEngine()
        # Устанавливаем первого игрока в соответствии с цветом MinMax (он ходит первым, если он белый)
        # В классических шашках белые ходят первыми. Если MinMax играет чёрными, то первым ходит DQN.
        if minmax_color == SideType.WHITE:
            env.current_player = SideType.WHITE
        else:
            env.current_player = SideType.BLACK  # DQN ходит первым

        minmax_ai = CheckersAI(env, depth=3)
        move_count = 0

        start_time = time.time()
        while not env.game_over:
            current = env.current_player
            move_start = time.time()
            if current == minmax_color:
                seq = minmax_ai.get_best_move(current)
                if not seq:
                    break
                for move in seq:
                    env.apply_move(move)
                move_time = time.time() - move_start
                stats["minmax_move_times"].append(move_time)
            else:  # DQN ход
                valid = env.get_possible_moves()
                if not valid:
                    break
                move, _ = dqn.select_action(env, valid, training=False)
                env.apply_move(move)
                move_time = time.time() - move_start
                stats["dqn_move_times"].append(move_time)
            move_count += 1
        game_time = time.time() - start_time

        # Подсчёт очков
        w_score = env.field.captured_white_score
        b_score = env.field.captured_black_score

        if minmax_color == SideType.WHITE:
            minmax_score = w_score
            dqn_score = b_score
        else:
            minmax_score = b_score
            dqn_score = w_score

        stats["minmax_total_score"] += minmax_score
        stats["dqn_total_score"] += dqn_score
        stats["total_moves"] += move_count

        winner = env.winner
        if winner == minmax_color:
            stats["minmax_wins"] += 1
            callback(f"  Победил MinMax (очки: MinMax={minmax_score}, DQN={dqn_score})")
        elif winner == dqn_color:
            stats["dqn_wins"] += 1
            callback(f"  Победил DQN (очки: MinMax={minmax_score}, DQN={dqn_score})")
        else:
            stats["draws"] += 1
            callback(f"  Ничья (очки: MinMax={minmax_score}, DQN={dqn_score})")
        callback(f"  Ходов в партии: {move_count}, время партии: {game_time:.2f} сек")

    # Вычисление средних
    total_games = num_games
    minmax_win_rate = stats["minmax_wins"] / total_games * 100
    dqn_win_rate = stats["dqn_wins"] / total_games * 100
    draw_rate = stats["draws"] / total_games * 100

    avg_minmax_score = stats["minmax_total_score"] / total_games
    avg_dqn_score = stats["dqn_total_score"] / total_games
    avg_moves = stats["total_moves"] / total_games

    avg_minmax_move_time = sum(stats["minmax_move_times"]) / len(stats["minmax_move_times"]) if stats[
        "minmax_move_times"] else 0
    avg_dqn_move_time = sum(stats["dqn_move_times"]) / len(stats["dqn_move_times"]) if stats["dqn_move_times"] else 0

    # Формируем строку с итогами
    result_str = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     ИТОГОВАЯ СТАТИСТИКА МАТЧА                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Всего партий:                 {total_games}                                    
    ║  ═══════════════════════════════════════════════════════════ ║
    ║  Победы MinMax:                {stats["minmax_wins"]}  ({minmax_win_rate:.1f}%)
    ║  Победы DQN:                   {stats["dqn_wins"]}  ({dqn_win_rate:.1f}%)
    ║  Ничьи:                        {stats["draws"]}  ({draw_rate:.1f}%)
    ║  ═══════════════════════════════════════════════════════════ ║
    ║  Средний счёт MinMax:          {avg_minmax_score:.2f} очков
    ║  Средний счёт DQN:             {avg_dqn_score:.2f} очков
    ║  ═══════════════════════════════════════════════════════════ ║
    ║  Среднее число ходов в партии: {avg_moves:.1f}
    ║  ═══════════════════════════════════════════════════════════ ║
    ║  Среднее время хода MinMax:    {avg_minmax_move_time * 1000:.2f} мс
    ║  Среднее время хода DQN:       {avg_dqn_move_time * 1000:.2f} мс
    ╚══════════════════════════════════════════════════════════════╝
    """
    callback(result_str)

    return {
        "minmax_wins": stats["minmax_wins"],
        "dqn_wins": stats["dqn_wins"],
        "draws": stats["draws"],
        "minmax_total_score": stats["minmax_total_score"],
        "dqn_total_score": stats["dqn_total_score"],
        "minmax_win_rate": minmax_win_rate,
        "dqn_win_rate": dqn_win_rate,
        "avg_minmax_score": avg_minmax_score,
        "avg_dqn_score": avg_dqn_score,
        "avg_moves": avg_moves,
        "avg_minmax_move_time_ms": avg_minmax_move_time * 1000,
        "avg_dqn_move_time_ms": avg_dqn_move_time * 1000
    }