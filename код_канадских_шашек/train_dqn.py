import numpy as np
from game_checkers import GameEngine, SideType, CheckerType, WHITE_QUEEN, BLACK_QUEEN
from dqn_agent import DQNAgent, board_to_tensor, move_to_embedding
from ai import CheckersAI

def train_dqn_vs_minimax(episodes=200, minimax_depth=1, model_path="dqn_model.pth"):
    agent = DQNAgent(learning_rate=1e-3, batch_size=32, target_update_freq=50,
                     epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.01)  # линейный распад
    for ep in range(episodes):
        env = GameEngine()
        # Чередование сторон
        dqn_side = SideType.WHITE if ep % 2 == 0 else SideType.BLACK
        minimax_side = dqn_side.opposite()
        minimax_ai = CheckersAI(env, depth=minimax_depth)
        state = board_to_tensor(env)
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 250:
            steps += 1
            current = env.current_player
            if current == dqn_side:
                valid = env.get_possible_moves()
                if not valid:
                    break
                move, move_idx = agent.select_action(env, valid, training=True)
                emb_all = np.array([move_to_embedding(m, current) for m in valid])
                old_state = state.copy()
                has_killed, killed_type = env.apply_move(move)
                reward = 0
                if has_killed:
                    reward = 3.0 if killed_type in (WHITE_QUEEN, BLACK_QUEEN) else 1.0
                # Shaping
                piece = env.field.type_at(move.to_x, move.to_y)
                if (dqn_side == SideType.WHITE and piece == CheckerType.WHITE_REGULAR):
                    reward += (env.field.height - move.to_y) * 0.02
                elif (dqn_side == SideType.BLACK and piece == CheckerType.BLACK_REGULAR):
                    reward += move.to_y * 0.02
                if env.game_over:
                    if env.winner == dqn_side:
                        reward += 100
                    elif env.winner == minimax_side:
                        reward -= 100
                done = env.game_over
                next_state = board_to_tensor(env)
                if not done:
                    next_valid = env.get_possible_moves()
                    next_embs = np.array([move_to_embedding(m, env.current_player) for m in next_valid])
                else:
                    next_embs = np.empty((0,9), dtype=np.float32)
                agent.store_transition(old_state, move_idx, emb_all, reward, next_state, next_embs, done)
                state = next_state
                total_reward += reward
                agent.learn()
            else:  # minimax ход
                seq = minimax_ai.get_best_move(current)
                if seq:
                    for move in seq:
                        env.apply_move(move)
                else:
                    break
        agent.update_epsilon()
        if steps % 50 == 0:
            print(
                f"  [эпизод {ep + 1}, шаг {steps}] продолжаем игру, фигур белых={env.field.white_count}, чёрных={env.field.black_count}")
        if (ep+1) % 20 == 0:
            agent.save_model(f"{model_path}.ep{ep+1}")
            print(f"Episode {ep+1}: total_reward={total_reward:.1f}, epsilon={agent.epsilon:.3f}")
    agent.save_model(model_path)
    print("Обучение завершено.")

if __name__ == "__main__":
    train_dqn_vs_minimax(episodes=150, minimax_depth=3)