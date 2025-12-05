#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# -----------------------
# Обработчики методов
# -----------------------

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def laplacian_sharpen(gray, ksize=3, alpha=1.0):
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    lap = cv2.convertScaleAbs(lap)
    sharp = cv2.addWeighted(gray, 1.0, lap, alpha, 0)
    return sharp

def unsharp_mask(gray, sigma=1.0, alpha=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = cv2.subtract(gray, blur)
    sharp = cv2.addWeighted(gray, 1.0, mask, alpha, 0)
    return sharp

def integral_image(img):
    return cv2.integral(img, sdepth=cv2.CV_64F)

def mean_in_window(integ, x, y, w, h):
    # интегральное изображение имеет размер (h+1, w+1)
    return (integ[y+h, x+w] - integ[y, x+w] - integ[y+h, x] + integ[y, x]) / (w * h)

def mean_var_maps(gray, win):
    h, w = gray.shape
    pad = win // 2
    # паддинг для удобства
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    integ = integral_image(padded)
    integ_sq = integral_image(np.square(padded, dtype=np.float64))
    mean = np.zeros_like(gray, dtype=np.float64)
    var = np.zeros_like(gray, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            m = mean_in_window(integ, j, i, win, win)
            s = mean_in_window(integ_sq, j, i, win, win)
            mean[i, j] = m
            var[i, j] = max(0.0, s - m * m)
    std = np.sqrt(var)
    return mean, std

def niblack_threshold(gray, win=15, k=-0.2):
    mean, std = mean_var_maps(gray, win)
    T = mean + k * std
    out = (gray > T).astype(np.uint8) * 255
    return out

def sauvola_threshold(gray, win=15, k=0.5, R=128):
    mean, std = mean_var_maps(gray, win)
    T = mean * (1 + k * (std / R - 1))
    out = (gray > T).astype(np.uint8) * 255
    return out

def adaptive_mean_threshold(gray, win=15, C=5):
    # Используем OpenCV для скорости
    out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, win | 1, C)
    return out

# -----------------------
# GUI
# -----------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР2 Вариант 13: ВЧ фильтры, локальные и адаптивные пороги")
        self.geometry("1000x600")
        self.img_bgr = None
        self.result = None
        self._build_ui()

    def _build_ui(self):
        control = ttk.Frame(self)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Button(control, text="Открыть изображение", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(control, text="Сохранить результат", command=self.save_image).pack(fill=tk.X, pady=2)

        ttk.Separator(control, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # Параметры
        self.method = tk.StringVar(value="laplacian")
        methods = [
            ("Лапласиан", "laplacian"),
            ("Unsharp", "unsharp"),
            ("Ниблэк", "niblack"),
            ("Саувола", "sauvola"),
            ("Адаптивный (средний)", "adaptive"),
        ]
        ttk.Label(control, text="Метод:").pack(anchor="w")
        for text, val in methods:
            ttk.Radiobutton(control, text=text, variable=self.method, value=val).pack(anchor="w")

        ttk.Label(control, text="Размер окна (нечётный):").pack(anchor="w", pady=(8, 0))
        self.win = tk.IntVar(value=15)
        ttk.Entry(control, textvariable=self.win, width=8).pack(anchor="w")

        ttk.Label(control, text="k (Ниблэк/Саувола):").pack(anchor="w", pady=(8, 0))
        self.k = tk.DoubleVar(value=-0.2)
        ttk.Entry(control, textvariable=self.k, width=8).pack(anchor="w")

        ttk.Label(control, text="R (Саувола):").pack(anchor="w", pady=(8, 0))
        self.R = tk.DoubleVar(value=128)
        ttk.Entry(control, textvariable=self.R, width=8).pack(anchor="w")

        ttk.Label(control, text="C (адаптивный):").pack(anchor="w", pady=(8, 0))
        self.C = tk.DoubleVar(value=5.0)
        ttk.Entry(control, textvariable=self.C, width=8).pack(anchor="w")

        ttk.Label(control, text="Alpha (лапл./unsharp):").pack(anchor="w", pady=(8, 0))
        self.alpha = tk.DoubleVar(value=1.0)
        ttk.Entry(control, textvariable=self.alpha, width=8).pack(anchor="w")

        ttk.Label(control, text="Sigma (unsharp):").pack(anchor="w", pady=(8, 0))
        self.sigma = tk.DoubleVar(value=1.0)
        ttk.Entry(control, textvariable=self.sigma, width=8).pack(anchor="w")

        ttk.Button(control, text="Применить", command=self.apply).pack(fill=tk.X, pady=10)

        # Панель изображений
        view = ttk.Frame(self)
        view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.label_orig = ttk.Label(view, text="Исходное", anchor="center")
        self.label_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.label_res = ttk.Label(view, text="Результат", anchor="center")
        self.label_res.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                                                     ("Все файлы", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение.")
            return
        self.img_bgr = img
        self.show_image(img, self.label_orig)
        self.result = None
        self.label_res.config(image="", text="Результат")

    def save_image(self):
        if self.result is None:
            messagebox.showwarning("Нет результата", "Сначала примените обработку.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")])
        if not path:
            return
        cv2.imwrite(path, self.result)
        messagebox.showinfo("Сохранено", f"Результат сохранён:\n{path}")

    def show_image(self, img_bgr, widget):
        # Масштабировать под виджет
        h, w = img_bgr.shape[:2]
        max_w = self.winfo_width() // 2 if self.winfo_width() > 0 else 500
        max_h = self.winfo_height() - 50 if self.winfo_height() > 0 else 500
        scale = min(max_w / w, max_h / h, 1.0)
        if scale <= 0:
            scale = 1.0
        img_disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(pil)
        widget.image = tk_img
        widget.config(image=tk_img, text="")

    def apply(self):
        if self.img_bgr is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте изображение.")
            return
        gray = to_gray(self.img_bgr)
        method = self.method.get()
        win = max(3, int(self.win.get()) | 1)  # нечётный
        try:
            if method == "laplacian":
                res = laplacian_sharpen(gray, ksize=3, alpha=float(self.alpha.get()))
            elif method == "unsharp":
                res = unsharp_mask(gray, sigma=float(self.sigma.get()), alpha=float(self.alpha.get()))
            elif method == "niblack":
                res = niblack_threshold(gray, win=win, k=float(self.k.get()))
            elif method == "sauvola":
                res = sauvola_threshold(gray, win=win, k=float(self.k.get()), R=float(self.R.get()))
            elif method == "adaptive":
                res = adaptive_mean_threshold(gray, win=win, C=float(self.C.get()))
            else:
                messagebox.showerror("Ошибка", "Неизвестный метод.")
                return
        except Exception as e:
            messagebox.showerror("Ошибка", f"При обработке возникла ошибка:\n{e}")
            return

        # Сохраняем и отображаем
        if res.ndim == 2:
            res_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
        else:
            res_bgr = res
        self.result = res_bgr
        self.show_image(res_bgr, self.label_res)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
