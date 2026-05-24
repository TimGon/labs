import tkinter as tk
from tkinter import messagebox
from constants import BOARD_WIDTH
from models import SideType
from game_checkers import Game
from ai import CheckersAI

from dqn_agent import DQNAgent
import threading

class GameGui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Канадские шашки')
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#2c3e50')
        self.current_frame = None
        self.turn_var = tk.StringVar(value="Ход: Белые")
        self.white_score_var = tk.StringVar(value="Очки белых: 0")
        self.black_score_var = tk.StringVar(value="Очки черных: 0")
        self.canvas = tk.Canvas()
        self.game = Game(self.canvas)
        self.show_main_menu()
        self.ai_side = None  # None – игра с человеком, иначе SideType.WHITE или BLACK
        self.ai = None

    def start_game_vs_dqn(self, human_side):
        """Человек против DQN (human_side = SideType.WHITE или BLACK)"""
        self.start_game(human_side=human_side, vs_dqn=True)

    def start_match_minmax_vs_dqn(self):
        """Запуск матча MinMax против DQN (без GUI, вывод в консоль)"""
        # Открываем новое окно для вывода результатов
        match_win = tk.Toplevel(self.root)
        match_win.title("Матч MinMax vs DQN")
        match_win.geometry("800x600")
        text_area = tk.Text(match_win, wrap=tk.WORD, font=("Courier", 11))
        text_area.pack(fill=tk.BOTH, expand=True)

        def run_match():
            from match import run_match_minmax_vs_dqn
            result = run_match_minmax_vs_dqn(num_games=10, callback=lambda msg: text_area.insert(tk.END, msg + "\n"))
            threading.Thread(target=run_match, daemon=True).start()

        threading.Thread(target=run_match, daemon=True).start()
    def clear_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = tk.Frame(self.root, bg='#2c3e50')
        self.current_frame.pack(fill='both', expand=True)

    def show_main_menu(self):
        self.clear_frame()
        frame = self.current_frame

        title = tk.Label(frame, text="Канадские шашки", font=("Arial", 36, "bold"),
                         fg='#ecf0f1', bg='#2c3e50')
        title.pack(pady=(50, 30))

        btn_frame = tk.Frame(frame, bg='#2c3e50')
        btn_frame.pack()

        btn_style = {'font': ("Arial", 16), 'width': 22, 'height': 2,
                     'relief': tk.FLAT, 'cursor': 'hand2'}

        tk.Button(btn_frame, text="Игра с человеком",
                  command=lambda: self.start_game(human_side=None),
                  bg='#27ae60', fg='white', **btn_style).pack(pady=10)
        tk.Button(btn_frame, text="Игра с ИИ (я белые)",
                  command=lambda: self.start_game(human_side=SideType.WHITE),
                  bg='#2980b9', fg='white', **btn_style).pack(pady=10)
        tk.Button(btn_frame, text="Игра с ИИ (я чёрные)",
                  command=lambda: self.start_game(human_side=SideType.BLACK),
                  bg='#2980b9', fg='white', **btn_style).pack(pady=10)
        tk.Button(btn_frame, text="Игра с нейросетью (DQN)", command=lambda: self.start_game_vs_dqn(SideType.WHITE)).pack(pady=10)
        tk.Button(btn_frame, text="Матч MinMax vs DQN", command=self.start_match_minmax_vs_dqn).pack(pady=10)
        tk.Button(btn_frame, text="Правила", command=self.show_rules,
                  bg='#8e44ad', fg='white', **btn_style).pack(pady=10)
        tk.Button(btn_frame, text="Выход", command=self.root.destroy,
                  bg='#c0392b', fg='white', **btn_style).pack(pady=10)

    def start_game(self, human_side=None, vs_dqn=False):
        self.clear_frame()
        frame = self.current_frame

        # Верхняя панель с информацией
        top_bar = tk.Frame(frame, bg='#34495e', height=50)
        top_bar.pack(fill='x', side='top')
        top_bar.pack_propagate(False)

        tk.Label(top_bar, textvariable=self.turn_var, font=("Arial", 18, "bold"),
                 bg='#34495e', fg='#ecf0f1').pack(pady=10)

        # Основной контейнер
        main_area = tk.Frame(frame, bg='#34495e')
        main_area.pack(fill='both', expand=True)

        # Левая часть (доска)
        board_frame = tk.Frame(main_area, bg='white', highlightbackground='#2c3e50', highlightthickness=2)
        board_frame.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(board_frame, bg='white', highlightthickness=0)
        self.canvas.pack(expand=True)

        # Правая панель
        right_panel = tk.Frame(main_area, bg='#34495e', width=300)
        right_panel.pack(side='right', fill='y')
        right_panel.pack_propagate(False)

        # Счёт
        score_frame = tk.Frame(right_panel, bg='#34495e')
        score_frame.pack(pady=20)

        tk.Label(score_frame, textvariable=self.white_score_var, font=("Arial", 14),
                 bg='#34495e', fg='#ecf0f1').pack(pady=5)
        tk.Label(score_frame, textvariable=self.black_score_var, font=("Arial", 14),
                 bg='#34495e', fg='#ecf0f1').pack(pady=5)

        # Кнопки
        btn_frame = tk.Frame(right_panel, bg='#34495e')
        btn_frame.pack(pady=20)
        btn_style = {'font': ("Arial", 12), 'width': 15, 'relief': tk.FLAT, 'cursor': 'hand2'}
        tk.Button(btn_frame, text="Новая игра", command=self.start_game,
                  bg='#27ae60', fg='white', **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Сдаться", command=self.surrender,
                  bg='#2980b9', fg='white', **btn_style).pack(pady=5)
        tk.Button(btn_frame, text="Главное меню", command=self.show_main_menu,
                  bg='#c0392b', fg='white', **btn_style).pack(pady=5)

        # Инициализация игры
        self.game = Game(self.canvas)
        self.game.draw()

        if vs_dqn:
            self.ai_side = human_side.opposite() if human_side is not None else None
            self.ai = None
            # Загружаем DQN агента
            self.dqn_agent = DQNAgent()
            self.dqn_agent.load_model("dqn_model.pth")  # предварительно обученная модель
            self.dqn_agent.epsilon = 0.0  # отключаем exploration при игре
        else:
            self.ai_side = human_side.opposite() if human_side is not None else None
            self.ai = CheckersAI(self.game, depth=3) if self.ai_side is not None else None
            self.dqn_agent = None

            # Привязка событий
        self.canvas.bind("<Motion>", self.game.mouse_move)
        self.canvas.bind("<Button-1>", self.on_mouse_click)  # новый обработчик

        # Обновление информации
        def update_info():
            if not self.game.game_over:
                if self.game.current_player == SideType.WHITE:
                    self.turn_var.set("Ход: Белые")
                else:
                    self.turn_var.set("Ход: Чёрные")
            else:
                winner = "Белые" if self.game.winner == SideType.WHITE else "Чёрные"
                self.turn_var.set(f"Победили {winner}")
            self.white_score_var.set(f"Очки белых: {self.game.field.captured_white_score}")
            self.black_score_var.set(f"Очки черных: {self.game.field.captured_black_score}")
            self.root.after(100, update_info)

        update_info()

        # Адаптивный размер
        def on_resize(event=None):
            w = board_frame.winfo_width()
            h = board_frame.winfo_height()
            size = min(w, h)
            self.canvas.config(width=size, height=size)
            self.cell_size = size // BOARD_WIDTH
            self.game.cell_size = self.cell_size
            self.game._init_images()
            self.game.draw()

        board_frame.bind('<Configure>', on_resize)

        # Если первым ходит компьютер – вызвать AI
        if self.ai_side is not None and self.game.current_player == self.ai_side:
            self.root.after(500, self.make_ai_move)  # небольшая задержка

        board_frame.bind('<Configure>', on_resize)

        # Обновление интерфейса
        def update_info():
            if not self.game.game_over:
                if self.game.current_player == SideType.WHITE:
                    self.turn_var.set("Ход: Белые")
                else:
                    self.turn_var.set("Ход: Чёрные")
            else:
                winner = "Белые" if self.game.winner == SideType.WHITE else "Чёрные"
                self.turn_var.set(f"Победили {winner}")
            self.white_score_var.set(f"Очки белых: {self.game.field.captured_white_score}")
            self.black_score_var.set(f"Очки черных: {self.game.field.captured_black_score}")
            self.root.after(100, update_info)

        update_info()
    def surrender(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите сдаться?"):
            winner = "Чёрные" if self.game.current_player == 1 else "Белые"
            messagebox.showinfo("Конец игры", f"{winner} выиграли!")
            self.start_game()  # перезапуск

    def show_rules(self):
        rules_win = tk.Toplevel(self.root)
        rules_win.title("Правила канадских шашек")
        rules_win.attributes('-fullscreen', True)
        rules_win.configure(bg='#f0f0f0')

        # Заголовок
        top = tk.Frame(rules_win, bg='#f0f0f0')
        top.pack(fill='x', padx=50, pady=20)
        tk.Label(top, text="Правила канадских шашек", font=("Arial", 24, "bold"),
                 bg='#f0f0f0', fg='#2c3e50').pack(side='left')
        tk.Button(top, text="✕", command=rules_win.destroy, font=("Arial", 16, "bold"),
                  bg="#e74c3c", fg="white", width=3, relief=tk.FLAT).pack(side='right')

        # Текст с прокруткой
        text_frame = tk.Frame(rules_win, bg='#f0f0f0')
        text_frame.pack(fill='both', expand=True, padx=50, pady=(0,30))
        scroll = tk.Scrollbar(text_frame)
        scroll.pack(side='right', fill='y')
        text = tk.Text(text_frame, wrap='word', font=("Arial", 14), padx=20, pady=20,
                       bg='white', fg='#2c3e50', yscrollcommand=scroll.set)
        text.pack(fill='both', expand=True)
        scroll.config(command=text.yview)

        rules = """
        ПРАВИЛА ИГРЫ В КАНАДСКИЕ ШАШКИ
        
            Основные положения:
            • Игра ведется на доске размером 12×12 клеток
            • Каждый игрок начинает с 30 шашками
            • Первый ход делают белые шашки
        
            Правила передвижения:
            1. Простая шашка ходит только вперед по диагонали на одну клетку.
        
            2. При достижении последней горизонтали простая шашка превращается в дамку.
        
            3. Дамка может ходить на любое количество клеток по диагонали как вперед, так и назад.
        
            Правила взятия:
            1. Взятие обязательно! Если есть возможность взять шашку противника, вы обязаны это сделать.
        
            2. При наличии нескольких вариантов взятия нужно выбрать тот, где будет взято наибольшее количество шашек противника.
        
            3. Взятие происходит через одну клетку по диагонали с перескоком через шашку противника.
        
            4. Взятые шашки снимаются с доски только после завершения полного хода.
        
            5. Простая шашка может бить как вперед, так и назад.
        
            Особые правила:
            • Если простая шашка во время взятия достигает последней горизонтали, но еще может продолжить взятие, она остается простой шашкой до завершения взятия.
        
            • "Турецкий удар" запрещен - нельзя дважды перепрыгивать через одну и ту же шашку противника.
        
            Окончание игры:
            Победа присуждается игроку, который:
            • Уничтожил все шашки противника
            • Или лишил их возможности хода ("запер")
        
            Подсчет очков:
            • За взятие простой шашки: 1 очко
            • За взятие дамки: 3 очка
        """
        text.insert('1.0', rules.strip())
        text.config(state='disabled')

        rules_win.bind('<Escape>', lambda e: rules_win.destroy())

    def on_mouse_click(self, event):
        if self.game.game_over or self.game.is_animating:
            return
        # Если сейчас ход компьютера – блокируем
        if self.ai_side is not None and self.game.current_player == self.ai_side:
            return
        self.game.mouse_down(event)
        # После хода человека проверить, не наступил ли ход компьютера
        if not self.game.game_over and self.ai_side is not None and self.game.current_player == self.ai_side:
            self.root.after(100, self.make_ai_move)

    def make_ai_move(self):
        if self.game.game_over or self.game.is_animating:
            return
        if self.ai_side is None or self.game.current_player != self.ai_side:
            return
        if self.ai is not None:
            best_seq = self.ai.get_best_move(self.ai_side)
            if best_seq:
                self.game.perform_full_move(best_seq)
        elif self.dqn_agent is not None:
            # DQN делает один ход (без полной последовательности, т.к. apply_move сам обработает продолжение)
            valid_moves = self.game.get_possible_moves()
            if not valid_moves:
                return
            move, _ = self.dqn_agent.select_action(self.game, valid_moves, training=False)
            self.game.make_move(move)  # make_move уже есть в классе Game
        # рекурсивный вызов если снова ход AI
        if not self.game.game_over and self.game.current_player == self.ai_side:
            self.root.after(100, self.make_ai_move)
    def run(self):
        self.root.mainloop()