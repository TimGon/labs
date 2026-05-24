import tkinter as tk
from tkinter import messagebox
import json
import hashlib

from gui import GameGui

def check_user(username: str, password: str) -> bool:
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return users.get(username) == hashed
    except FileNotFoundError:
        return False

def register_user(username: str, password: str) -> bool:
    try:
        with open('users.json', 'r') as f:
            users = json.load(f)
    except FileNotFoundError:
        users = {}
    if username in users:
        return False
    users[username] = hashlib.sha256(password.encode()).hexdigest()
    with open('users.json', 'w') as f:
        json.dump(users, f)
    return True

class AuthApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Авторизация')
        self.center_window(400, 500)
        self.root.resizable(False, False)
        self.root.configure(bg='#2c3e50')
        self.show_auth()

    def center_window(self, width, height):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def show_auth(self):
        self.clear()
        main = tk.Frame(self.root, bg='#2c3e50')
        main.pack(expand=True)

        self._make_label(main, 'Авторизация', 24).pack(pady=(30,20))

        entry_frame = tk.Frame(main, bg='#2c3e50')
        entry_frame.pack(pady=20)

        self._make_label(entry_frame, 'Имя пользователя').pack(pady=5)
        username = tk.Entry(entry_frame, **self._entry_style())
        username.pack(pady=5)

        self._make_label(entry_frame, 'Пароль').pack(pady=5)
        password = tk.Entry(entry_frame, show='*', **self._entry_style())
        password.pack(pady=5)

        btn_frame = tk.Frame(main, bg='#2c3e50')
        btn_frame.pack(pady=30)

        tk.Button(btn_frame, text='Войти', bg='#27ae60', fg='white',
                  command=lambda: self.login(username.get(), password.get()),
                  **self._btn_style()).pack(pady=10)
        tk.Button(btn_frame, text='Регистрация', bg='#2980b9', fg='white',
                  command=self.show_reg, **self._btn_style()).pack(pady=10)
        tk.Button(btn_frame, text='Выход', bg='#c0392b', fg='white',
                  command=self.root.destroy, **self._btn_style()).pack(pady=10)

    def show_reg(self):
        self.clear()
        main = tk.Frame(self.root, bg='#2c3e50')
        main.pack(expand=True)

        self._make_label(main, 'Регистрация', 24).pack(pady=(30,20))

        entry_frame = tk.Frame(main, bg='#2c3e50')
        entry_frame.pack(pady=20)

        self._make_label(entry_frame, 'Придумайте имя пользователя').pack(pady=5)
        username = tk.Entry(entry_frame, **self._entry_style())
        username.pack(pady=5)

        self._make_label(entry_frame, 'Придумайте пароль').pack(pady=5)
        password = tk.Entry(entry_frame, show='*', **self._entry_style())
        password.pack(pady=5)

        self._make_label(entry_frame, 'Повторите пароль').pack(pady=5)
        confirm = tk.Entry(entry_frame, show='*', **self._entry_style())
        confirm.pack(pady=5)

        btn_frame = tk.Frame(main, bg='#2c3e50')
        btn_frame.pack(pady=30)

        tk.Button(btn_frame, text='Зарегистрироваться', bg='#27ae60', fg='white',
                  command=lambda: self.register(username.get(), password.get(), confirm.get()),
                  **self._btn_style()).pack(pady=10)
        tk.Button(btn_frame, text='Назад', bg='#2980b9', fg='white',
                  command=self.show_auth, **self._btn_style()).pack(pady=10)

    def login(self, username, password):
        if not username or not password:
            messagebox.showerror("Ошибка", "Заполните все поля")
            return
        if check_user(username, password):
            self.root.destroy()
            GameGui().run()
        else:
            messagebox.showerror("Ошибка", "Неверное имя пользователя или пароль")

    def register(self, username, password, confirm):
        if not username or not password or not confirm:
            messagebox.showerror("Ошибка", "Заполните все поля")
            return
        if len(username) < 3:
            messagebox.showerror("Ошибка", "Имя пользователя должно быть не менее 3 символов")
            return
        if len(password) < 6:
            messagebox.showerror("Ошибка", "Пароль должен быть не менее 6 символов")
            return
        if password != confirm:
            messagebox.showerror("Ошибка", "Пароли не совпадают")
            return
        if register_user(username, password):
            messagebox.showinfo("Успех", "Регистрация успешна!")
            self.show_auth()
        else:
            messagebox.showerror("Ошибка", "Пользователь с таким именем уже существует")

    @staticmethod
    def _make_label(parent, text, size=10):
        return tk.Label(parent, text=text, font=('Arial', size, 'bold' if size>12 else 'normal'),
                        bg='#2c3e50', fg='#ecf0f1')

    @staticmethod
    def _entry_style():
        return {'bg': '#ecf0f1', 'fg': '#2c3e50', 'font': ('Arial', 12),
                'relief': tk.FLAT, 'width': 25}

    @staticmethod
    def _btn_style():
        return {'font': ('Arial', 12), 'relief': tk.FLAT, 'cursor': 'hand2',
                'width': 20, 'height': 2, 'activeforeground': 'white'}

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    AuthApp().run()