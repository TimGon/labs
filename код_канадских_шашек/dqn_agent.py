import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List

from models import Move, SideType, CheckerType
from game_checkers import GameEngine

# Преобразование доски в тензор
def board_to_tensor(engine: GameEngine) -> np.ndarray:
    h, w = engine.field.height, engine.field.width
    channels = 6
    tensor = np.zeros((channels, h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            piece = engine.field.type_at(x, y)
            if piece == CheckerType.WHITE_REGULAR:
                tensor[1, y, x] = 1.0
            elif piece == CheckerType.WHITE_QUEEN:
                tensor[2, y, x] = 1.0
            elif piece == CheckerType.BLACK_REGULAR:
                tensor[3, y, x] = 1.0
            elif piece == CheckerType.BLACK_QUEEN:
                tensor[4, y, x] = 1.0
    tensor[5, :, :] = 1.0 if engine.current_player == SideType.WHITE else 0.0
    return tensor

def move_to_embedding(move: Move, side: SideType) -> np.ndarray:
    from_x_n = move.from_x / 11.0
    from_y_n = move.from_y / 11.0
    to_x_n = move.to_x / 11.0
    to_y_n = move.to_y / 11.0
    dx = (move.to_x - move.from_x) / 11.0
    dy = (move.to_y - move.from_y) / 11.0
    length = np.sqrt(dx*dx + dy*dy)
    length_n = length / 15.5
    is_capture = 1.0 if abs(move.to_x - move.from_x) > 1 else 0.0
    side_val = 1.0 if side == SideType.WHITE else 0.0
    return np.array([from_x_n, from_y_n, to_x_n, to_y_n, is_capture, dx, dy, length_n, side_val], dtype=np.float32)

# Нейронная сеть
class DQNNetwork(nn.Module):
    def __init__(self, state_channels=6, board_size=12, action_emb_size=9, hidden_size=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        conv_out_size = 128 * board_size * board_size
        self.fc_state = nn.Linear(conv_out_size, hidden_size)
        self.fc_joint = nn.Sequential(
            nn.Linear(hidden_size + action_emb_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state_tensor, action_embeddings):
        # action_embeddings: (batch, num_actions, emb_dim)
        batch_size, num_actions, emb_dim = action_embeddings.shape
        conv_out = self.conv(state_tensor)                         # (batch, 128, h, w)
        conv_out = conv_out.view(batch_size, -1)                  # (batch, 128*h*w)
        state_feat = self.fc_state(conv_out)                      # (batch, hidden)
        state_feat_expanded = state_feat.unsqueeze(1).expand(-1, num_actions, -1)  # (batch, num_actions, hidden)
        joint = torch.cat([state_feat_expanded, action_embeddings], dim=2)         # (batch, num_actions, hidden+emb)
        q_values = self.fc_joint(joint).squeeze(-1)               # (batch, num_actions)
        return q_values

# Буфер воспроизведения
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, action_embs_all, reward, next_state, next_action_embs_all, done):
        self.buffer.append((state, action_idx, action_embs_all, reward, next_state, next_action_embs_all, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, action_idxs, action_embs_all, rewards, next_states, next_action_embs_all, dones = zip(*batch)
        return (np.array(states), action_idxs, list(action_embs_all),
                np.array(rewards), np.array(next_states), list(next_action_embs_all), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# DQN агент
class DQNAgent:
    def __init__(self, learning_rate=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.995, batch_size=64, target_update_freq=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def select_action(self, engine: GameEngine, valid_moves: List[Move], training=True):
        if training and np.random.random() < self.epsilon:
            idx = np.random.randint(len(valid_moves))
            return valid_moves[idx], idx

        state_tensor = torch.tensor(board_to_tensor(engine), dtype=torch.float32).unsqueeze(0).to(self.device)
        side = engine.current_player
        embeddings = np.array([move_to_embedding(m, side) for m in valid_moves])  # (num_actions, emb_dim)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, num_actions, emb_dim)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor, emb_tensor).squeeze(0)  # (num_actions,)
        best_idx = torch.argmax(q_values).item()
        return valid_moves[best_idx], best_idx

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action_idx, action_embs_all, reward, next_state, next_action_embs_all, done):
        self.replay_buffer.push(state, action_idx, action_embs_all, reward, next_state, next_action_embs_all, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        (states, action_idxs, action_embs_all,
         rewards, next_states, next_action_embs_all, dones) = self.replay_buffer.sample(self.batch_size)

        # Преобразуем в тензоры
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)           # (batch, C, H, W)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)         # (batch,)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Для каждого перехода у нас есть список эмбеддингов всех действий.
        # Нужно создать тензор (batch, max_num_actions, emb_dim) и маску.
        # Т.к. количество действий может отличаться, обрезаем/паддим до максимального.
        max_num_actions = max(emb.shape[0] for emb in action_embs_all)
        emb_dim = action_embs_all[0].shape[1]

        action_embs_batch = np.zeros((self.batch_size, max_num_actions, emb_dim), dtype=np.float32)
        for i, emb in enumerate(action_embs_all):
            action_embs_batch[i, :emb.shape[0]] = emb
        action_embs_t = torch.tensor(action_embs_batch, dtype=torch.float32).to(self.device)

        # Тензор для next state
        next_max_actions = max(emb.shape[0] for emb in next_action_embs_all)
        next_embs_batch = np.zeros((self.batch_size, next_max_actions, emb_dim), dtype=np.float32)
        for i, emb in enumerate(next_action_embs_all):
            next_embs_batch[i, :emb.shape[0]] = emb
        next_embs_t = torch.tensor(next_embs_batch, dtype=torch.float32).to(self.device)

        # Вычисляем Q(s,a) для всех a
        q_values_all = self.policy_net(states_t, action_embs_t)  # (batch, max_num_actions)
        # Выбираем Q для реально выбранных действий
        action_idxs_t = torch.tensor(action_idxs, dtype=torch.long).to(self.device)
        q_selected = q_values_all.gather(1, action_idxs_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # Вычисляем max Q(s',a') с помощью целевой сети
        with torch.no_grad():
            next_q_all = self.target_net(next_states_t, next_embs_t)   # (batch, next_max_actions)
            max_next_q, _ = next_q_all.max(dim=1)                      # (batch,)
            target = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = F.mse_loss(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path="dqn_model.pth"):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path="dqn_model.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())