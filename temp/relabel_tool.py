"""
relabel_tool.py — Công cụ giao diện đồ họa (GUI) xem và gán lại nhãn dữ liệu ảnh giao thông.
Được xây dựng từ cấu hình đường dẫn tại coral_focal.py (L738-L739).

Tính năng chính:
  1. Đọc và tải tập dữ liệu từ file CSV (chứa cột filename, phan_loai) và thư mục chứa ảnh.
  2. Hiển thị ảnh kèm nhãn hiện tại, tên file, chỉ số dòng.
  3. Lọc danh sách ảnh theo từng mức nhãn (Tất cả, Nhãn 1..5, hoặc nhãn đã chỉnh sửa).
  4. Đánh lại nhãn nhanh bằng Nút bấm hoặc Phím tắt ('1', '2', '3', '4', '5').
  5. Tự động chuyển tiếp sang ảnh kế tiếp sau khi gán nhãn.
  6. Lưu thay đổi đè lên file CSV hiện tại (có tạo file backup .bak) hoặc xuất thành file CSV mới.
"""

import os
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk

# Cấu hình đường dẫn mặc định từ coral_focal.py (L738-L739)
CSV_FILE = r'C:\Users\levie\Downloads\traffic_final_windows_order.csv'
IMAGE_DIR = r'C:\Users\levie\Downloads\archive\images'

# Tên các class mức độ tắc nghẽn
CLASS_NAMES = {
    1: "1 - Thông thoáng",
    2: "2 - Mật độ trung bình",
    3: "3 - Đông đúc",
    4: "4 - Ún tắc",
    5: "5 - Kẹt xe nghiêm trọng"
}

# Màu sắc phân biệt các nhãn (Dark theme accent)
CLASS_COLORS = {
    1: "#2ecc71",  # Xanh lá
    2: "#3498db",  # Xanh dương
    3: "#f1c40f",  # Vàng
    4: "#e67e22",  # Cam
    5: "#e74c3c"   # Đỏ
}


class RelabelApp:
    def __init__(self, root, csv_path, img_dir):
        self.root = root
        self.root.title("Traffic Image Relabeling Tool — Antigravity NCKH")
        self.root.geometry("1100x780")
        self.root.minsize(900, 650)

        # Trạng thái dữ liệu
        self.csv_path = tk.StringVar(value=csv_path)
        self.img_dir = tk.StringVar(value=img_dir)
        
        self.df = None
        self.original_labels = []  # Lưu nhãn gốc để theo dõi thay đổi
        self.modified_mask = []    # Đánh dấu các bản ghi đã được sửa nhãn
        self.filtered_indices = [] # Danh sách index trong df tương ứng với bộ lọc
        self.current_filter_idx = 0 # Chỉ số hiện tại trong filtered_indices
        self.history = []          # Lich sử thay đổi để Undo (ctrl+z)

        # Cấu hình giao diện Tkinter Style
        self.setup_styles()
        
        # Tạo giao diện
        self.create_widgets()

        # Đăng ký phím tắt
        self.bind_events()

        # Tải dữ liệu ban đầu nếu đường dẫn tồn tại
        if os.path.exists(self.csv_path.get()):
            self.load_dataset()
        else:
            self.update_status("Sẵn sàng. Vui lòng chọn đường dẫn CSV và thư mục ảnh hợp lệ để tải dữ liệu.")

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom Colors
        self.bg_color = "#1e1e2e"
        self.fg_color = "#cdd6f4"
        self.card_bg = "#252538"
        
        self.root.configure(bg=self.bg_color)
        
        # TTK styles
        self.style.configure(".", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Card.TFrame", background=self.card_bg, relief="flat")
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("Card.TLabel", background=self.card_bg, foreground=self.fg_color)
        self.style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#89b4fa")
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#cba6f7")
        self.style.configure("Status.TLabel", font=("Segoe UI", 9, "italic"), foreground="#a6adc8")
        
        self.style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=5)
        self.style.configure("Nav.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Save.TButton", font=("Segoe UI", 10, "bold"), background="#a6e3a1", foreground="#11111b")

    def create_widgets(self):
        # 1. Top Panel: Đường dẫn & Đóng/Tải lại
        top_frame = ttk.LabelFrame(self.root, text=" 📁 Cấu hình Đường dẫn Dữ liệu ", padding=10)
        top_frame.pack(fill="x", padx=15, pady=8)

        # CSV row
        ttk.Label(top_frame, text="Tệp CSV:").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        csv_entry = ttk.Entry(top_frame, textvariable=self.csv_path, width=65)
        csv_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=3)
        ttk.Button(top_frame, text="Browse CSV...", command=self.browse_csv).grid(row=0, column=2, padx=5, pady=3)

        # Image Dir row
        ttk.Label(top_frame, text="Thư mục Ảnh:").grid(row=1, column=0, sticky="w", padx=5, pady=3)
        img_entry = ttk.Entry(top_frame, textvariable=self.img_dir, width=65)
        img_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        ttk.Button(top_frame, text="Browse Ảnh...", command=self.browse_img_dir).grid(row=1, column=2, padx=5, pady=3)

        # Load button
        ttk.Button(top_frame, text="🔄 Tải Dữ Liệu", command=self.load_dataset).grid(row=0, column=3, rowspan=2, padx=10, pady=3, sticky="nsew")
        top_frame.columnconfigure(1, weight=1)

        # 2. Main Content Split Panel (Left: Image Display, Right: Controls & Stats)
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=15, pady=5)

        # ── Left Frame: Frame Xem Ảnh ─────────────────────────────────
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        # Image Canvas Wrapper
        self.image_container = ttk.Frame(left_frame, style="Card.TFrame", padding=10)
        self.image_container.pack(fill="both", expand=True)

        self.image_canvas = tk.Canvas(self.image_container, bg="#11111b", highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True)
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        # Bottom Bar under image: Navigation
        nav_frame = ttk.Frame(left_frame, padding=5)
        nav_frame.pack(fill="x", pady=8)

        self.btn_prev = ttk.Button(nav_frame, text="⬅️  Trước [Left/A]", style="Nav.TButton", command=self.prev_image)
        self.btn_prev.pack(side="left", padx=5)

        self.lbl_page = ttk.Label(nav_frame, text="Ảnh 0 / 0", font=("Segoe UI", 11, "bold"))
        self.lbl_page.pack(side="left", expand=True)

        # Jump to index
        jump_frame = ttk.Frame(nav_frame)
        jump_frame.pack(side="right", padx=5)
        ttk.Label(jump_frame, text="Nhảy tới index:").pack(side="left", padx=2)
        self.ent_jump = ttk.Entry(jump_frame, width=6)
        self.ent_jump.pack(side="left", padx=2)
        self.ent_jump.bind("<Return>", self.jump_to_index)
        ttk.Button(jump_frame, text="Go", width=4, command=self.jump_to_index).pack(side="left", padx=2)

        self.btn_next = ttk.Button(nav_frame, text="Sau [Right/D] ➡️", style="Nav.TButton", command=self.next_image)
        self.btn_next.pack(side="right", padx=5)

        # ── Right Frame: Filter, Label Buttons, Metadata & Stats ──────
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # A. Filter Box
        filter_box = ttk.LabelFrame(right_frame, text=" 🔍 Bộ Lọc Nhãn ", padding=10)
        filter_box.pack(fill="x", padx=5, pady=5)

        ttk.Label(filter_box, text="Lọc theo nhãn:").pack(side="left", padx=5)
        self.cmb_filter = ttk.Combobox(
            filter_box,
            state="readonly",
            values=[
                "Tất cả nhãn (All)",
                "Nhãn 1 - Thông thoáng",
                "Nhãn 2 - Mật độ trung bình",
                "Nhãn 3 - Đông đúc",
                "Nhãn 4 - Ùn tắc",
                "Nhãn 5 - Kẹt xe nghiêm trọng",
                "⚠️ Các ảnh đã chỉnh sửa nhãn"
            ],
            width=26
        )
        self.cmb_filter.current(0)
        self.cmb_filter.pack(side="left", padx=5, fill="x", expand=True)
        self.cmb_filter.bind("<<ComboboxSelected>>", self.on_filter_changed)

        # B. Metadata Box
        meta_box = ttk.LabelFrame(right_frame, text=" 📌 Thông Tin Mẫu Hiện Tại ", padding=10)
        meta_box.pack(fill="x", padx=5, pady=5)

        self.lbl_filename = ttk.Label(meta_box, text="File: -", font=("Consolas", 10, "bold"), foreground="#f9e2af")
        self.lbl_filename.pack(anchor="w", pady=2)

        self.lbl_current_label = ttk.Label(meta_box, text="Nhãn hiện tại: -", font=("Segoe UI", 12, "bold"))
        self.lbl_current_label.pack(anchor="w", pady=2)

        self.lbl_orig_label = ttk.Label(meta_box, text="Nhãn ban đầu: -", style="Status.TLabel")
        self.lbl_orig_label.pack(anchor="w", pady=2)

        self.lbl_mod_status = ttk.Label(meta_box, text="Trạng thái: Chưa chỉnh sửa", style="Status.TLabel")
        self.lbl_mod_status.pack(anchor="w", pady=2)

        # C. Relabel Control Box
        relabel_box = ttk.LabelFrame(right_frame, text=" 🏷️ Gán Lại Nhãn (Phím tắt 1 - 5) ", padding=10)
        relabel_box.pack(fill="x", padx=5, pady=5)

        self.label_buttons = []
        for cls_val in range(1, 6):
            btn = tk.Button(
                relabel_box,
                text=f"[{cls_val}]  {CLASS_NAMES[cls_val]}",
                font=("Segoe UI", 10, "bold"),
                bg=CLASS_COLORS[cls_val],
                fg="#ffffff",
                activebackground="#ffffff",
                activeforeground="#000000",
                relief="flat",
                pady=6,
                cursor="hand2",
                command=lambda c=cls_val: self.relabel_current(c)
            )
            btn.pack(fill="x", pady=3)
            self.label_buttons.append(btn)

        # Auto Advance Checkbox
        self.var_auto_advance = tk.BooleanVar(value=True)
        chk_auto = ttk.Checkbutton(relabel_box, text="Tự động sang ảnh tiếp theo sau khi gán nhãn", variable=self.var_auto_advance)
        chk_auto.pack(anchor="w", pady=6)

        # Undo Button
        ttk.Button(relabel_box, text="↩️ Hoàn tác đổi nhãn vừa rồi (Ctrl+Z)", command=self.undo_relabel).pack(fill="x", pady=2)

        # D. Statistics Box
        self.stats_box = ttk.LabelFrame(right_frame, text=" 📊 Thống Kê Phân Phối Nhãn ", padding=10)
        self.stats_box.pack(fill="x", padx=5, pady=5)

        self.lbl_stats = {}
        for c in range(1, 6):
            lbl = ttk.Label(self.stats_box, text=f"Nhãn {c}: 0 mẫu (0.0%)")
            lbl.pack(anchor="w", pady=1)
            self.lbl_stats[c] = lbl

        # E. Save Box
        save_box = ttk.Frame(right_frame, padding=5)
        save_box.pack(fill="x", padx=5, pady=10)

        self.btn_save = ttk.Button(save_box, text="💾 GHI ĐÈ LƯU FILE CSV", style="Save.TButton", command=self.save_csv)
        self.btn_save.pack(side="left", expand=True, fill="x", padx=3)

        ttk.Button(save_box, text="📂 Lưu thành CSV mới...", command=self.save_csv_as).pack(side="right", expand=True, fill="x", padx=3)

        # 3. Bottom Status Bar
        self.status_bar = ttk.Label(self.root, text="Sẵn sàng", style="Status.TLabel", padding=5, relief="sunken")
        self.status_bar.pack(fill="x", side="bottom")

    def bind_events(self):
        # Hotkeys cho số 1 -> 5
        for i in range(1, 6):
            self.root.bind(str(i), lambda event, c=i: self.relabel_current(c))

        # Điều hướng phím mũi tên & A/D
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("a", lambda e: self.prev_image())
        self.root.bind("A", lambda e: self.prev_image())
        self.root.bind("d", lambda e: self.next_image())
        self.root.bind("D", lambda e: self.next_image())

        # Ctrl+Z để Undo
        self.root.bind("<Control-z>", lambda e: self.undo_relabel())
        self.root.bind("<Control-Z>", lambda e: self.undo_relabel())

    def update_status(self, text):
        self.status_bar.config(text=text)

    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="Chọn file CSV chứa dữ liệu nhãn",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.csv_path.set(filename)

    def browse_img_dir(self):
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        if folder:
            self.img_dir.set(folder)

    def load_dataset(self):
        path = self.csv_path.get()
        img_directory = self.img_dir.get()

        if not os.path.exists(path):
            messagebox.showerror("Lỗi đường dẫn", f"Không tìm thấy file CSV tại:\n{path}")
            return

        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Lỗi đọc CSV", f"Không thể đọc tệp CSV:\n{e}")
            return

        # Kiểm tra cột bắt buộc
        if 'filename' not in self.df.columns or 'phan_loai' not in self.df.columns:
            messagebox.showerror("Lỗi định dạng", "Tệp CSV phải chứa ít nhất 2 cột: 'filename' và 'phan_loai'.")
            return

        # Ép kiểu dữ liệu
        self.df['phan_loai'] = pd.to_numeric(self.df['phan_loai'], errors='coerce').fillna(1).astype(int)
        self.original_labels = self.df['phan_loai'].values.copy()
        self.modified_mask = [False] * len(self.df)
        self.history.clear()

        self.apply_filter()
        self.update_stats()
        self.update_status(f"Đã tải thành công dataset với {len(self.df)} mẫu ảnh từ {os.path.basename(path)}.")

    def apply_filter(self):
        if self.df is None or len(self.df) == 0:
            self.filtered_indices = []
            self.current_filter_idx = 0
            self.render_current_item()
            return

        filter_sel = self.cmb_filter.current()

        if filter_sel == 0:  # Tất cả
            self.filtered_indices = list(range(len(self.df)))
        elif 1 <= filter_sel <= 5:  # Nhãn 1 đến 5
            target_class = filter_sel
            self.filtered_indices = self.df.index[self.df['phan_loai'] == target_class].tolist()
        elif filter_sel == 6:  # Các ảnh đã sửa nhãn
            self.filtered_indices = [i for i, mod in enumerate(self.modified_mask) if mod]

        self.current_filter_idx = 0
        self.render_current_item()

    def on_filter_changed(self, event=None):
        self.apply_filter()

    def render_current_item(self):
        if not self.filtered_indices or self.df is None:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(
                self.image_canvas.winfo_width() // 2 or 250,
                self.image_canvas.winfo_height() // 2 or 200,
                text="Không có mẫu ảnh nào phù hợp bộ lọc!",
                fill="#f38ba8",
                font=("Segoe UI", 14, "bold")
            )
            self.lbl_page.config(text="Ảnh 0 / 0")
            self.lbl_filename.config(text="File: -")
            self.lbl_current_label.config(text="Nhãn hiện tại: -", foreground=self.fg_color)
            self.lbl_orig_label.config(text="Nhãn ban đầu: -")
            self.lbl_mod_status.config(text="Trạng thái: -")
            return

        total_filtered = len(self.filtered_indices)
        if self.current_filter_idx >= total_filtered:
            self.current_filter_idx = total_filtered - 1
        elif self.current_filter_idx < 0:
            self.current_filter_idx = 0

        row_idx = self.filtered_indices[self.current_filter_idx]
        row = self.df.iloc[row_idx]

        filename = str(row['filename'])
        curr_label = int(row['phan_loai'])
        orig_label = int(self.original_labels[row_idx])
        is_mod = self.modified_mask[row_idx]

        # Metadata display
        self.lbl_page.config(text=f"Ảnh {self.current_filter_idx + 1} / {total_filtered}  (Dòng CSV: {row_idx + 1})")
        self.lbl_filename.config(text=f"File: {filename}")
        
        lbl_text = CLASS_NAMES.get(curr_label, f"Nhãn {curr_label}")
        lbl_color = CLASS_COLORS.get(curr_label, "#ffffff")
        self.lbl_current_label.config(text=f"Nhãn hiện tại: {lbl_text}", foreground=lbl_color)
        
        orig_text = CLASS_NAMES.get(orig_label, f"Nhãn {orig_label}")
        self.lbl_orig_label.config(text=f"Nhãn ban đầu: {orig_text}")

        if is_mod:
            self.lbl_mod_status.config(text="Trạng thái: ✏️ Đã chỉnh sửa nhãn", foreground="#f9e2af")
        else:
            self.lbl_mod_status.config(text="Trạng thái: Gốc (Chưa chỉnh sửa)", foreground="#a6adc8")

        # Load image
        img_path = os.path.join(self.img_dir.get(), filename)
        self.current_pil_img = None

        if os.path.exists(img_path):
            try:
                self.current_pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                self.display_canvas_error(f"Lỗi đọc ảnh ({filename}):\n{e}")
                return
        else:
            self.display_canvas_error(f"⚠️ không tìm thấy tệp ảnh:\n{img_path}")
            return

        self.draw_image_on_canvas()

    def display_canvas_error(self, message):
        self.image_canvas.delete("all")
        w = max(self.image_canvas.winfo_width(), 300)
        h = max(self.image_canvas.winfo_height(), 200)
        self.image_canvas.create_text(
            w // 2, h // 2,
            text=message,
            fill="#f38ba8",
            font=("Segoe UI", 11, "bold"),
            justify="center"
        )

    def draw_image_on_canvas(self):
        if not hasattr(self, 'current_pil_img') or self.current_pil_img is None:
            return

        cw = self.image_canvas.winfo_width()
        ch = self.image_canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        img_w, img_h = self.current_pil_img.size
        scale = min(cw / img_w, ch / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = self.current_pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.image_canvas.delete("all")
        # Center image
        x_pos = (cw - new_w) // 2
        y_pos = (ch - new_h) // 2
        self.image_canvas.create_image(x_pos, y_pos, anchor="nw", image=self.tk_img)

    def on_canvas_resize(self, event):
        self.draw_image_on_canvas()

    def prev_image(self):
        if self.filtered_indices and self.current_filter_idx > 0:
            self.current_filter_idx -= 1
            self.render_current_item()

    def next_image(self):
        if self.filtered_indices and self.current_filter_idx < len(self.filtered_indices) - 1:
            self.current_filter_idx += 1
            self.render_current_item()

    def jump_to_index(self, event=None):
        val = self.ent_jump.get().strip()
        if not val.isdigit():
            return
        idx = int(val) - 1
        if 0 <= idx < len(self.filtered_indices):
            self.current_filter_idx = idx
            self.render_current_item()
            self.ent_jump.delete(0, tk.END)

    def relabel_current(self, new_label):
        if not self.filtered_indices or self.df is None:
            return

        row_idx = self.filtered_indices[self.current_filter_idx]
        old_label = int(self.df.at[row_idx, 'phan_loai'])

        if old_label == new_label:
            if self.var_auto_advance.get():
                self.next_image()
            return

        # Đổi nhãn & lưu lịch sử để Undo
        self.df.at[row_idx, 'phan_loai'] = new_label
        self.modified_mask[row_idx] = True
        self.history.append((row_idx, old_label, new_label))

        filename = self.df.at[row_idx, 'filename']
        self.update_status(f"Đã đổi nhãn tệp '{filename}': Class {old_label} ➔ Class {new_label}")

        self.update_stats()

        # Render lại giao diện
        self.render_current_item()

        # Tự động sang ảnh tiếp theo nếu chọn checkbox
        if self.var_auto_advance.get():
            self.next_image()

    def undo_relabel(self):
        if not self.history:
            self.update_status("Không có thao tác nào để hoàn tác (Undo).")
            return

        row_idx, old_label, new_label = self.history.pop()
        self.df.at[row_idx, 'phan_loai'] = old_label
        
        # Kiểm tra xem có về lại nhãn gốc hay không
        if old_label == self.original_labels[row_idx]:
            self.modified_mask[row_idx] = False

        self.update_stats()
        self.update_status(f"Hoàn tác: Trả tệp '{self.df.at[row_idx, 'filename']}' về Class {old_label}")
        
        # Nhảy về lại item đó để xem
        if row_idx in self.filtered_indices:
            self.current_filter_idx = self.filtered_indices.index(row_idx)
        self.render_current_item()

    def update_stats(self):
        if self.df is None:
            return

        counts = self.df['phan_loai'].value_counts()
        total = len(self.df)
        mod_count = sum(self.modified_mask)

        for c in range(1, 6):
            cnt = counts.get(c, 0)
            pct = (cnt / total * 100) if total > 0 else 0.0
            self.lbl_stats[c].config(text=f"Nhãn {c} ({CLASS_NAMES[c]}): {cnt} mẫu ({pct:.1f}%)")

        self.stats_box.config(text=f" 📊 Thống Kê ({total} mẫu | ✏️ Đã sửa {mod_count} mẫu) ")

    def save_csv(self):
        if self.df is None:
            return

        path = self.csv_path.get()
        if not path:
            return

        # Tạo file backup .bak
        if os.path.exists(path):
            backup_path = path + ".bak"
            try:
                shutil.copy2(path, backup_path)
            except Exception as e:
                print(f"Cảnh báo: Không thể tạo backup file: {e}")

        try:
            self.df.to_csv(path, index=False)
            mod_count = sum(self.modified_mask)
            messagebox.showinfo("Lưu thành công", f"Đã ghi đè thành công dữ liệu nhãn mới vào:\n{path}\n(Tổng cộng {mod_count} mẫu được cập nhật, đã tạo sao lưu {os.path.basename(path)}.bak)")
            self.update_status(f"Đã lưu thành công tệp {os.path.basename(path)}.")
        except Exception as e:
            messagebox.showerror("Lỗi lưu file", f"Không thể lưu file CSV:\n{e}")

    def save_csv_as(self):
        if self.df is None:
            return

        new_path = filedialog.asksaveasfilename(
            title="Lưu file CSV mới",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not new_path:
            return

        try:
            self.df.to_csv(new_path, index=False)
            messagebox.showinfo("Lưu thành công", f"Đã xuất dữ liệu nhãn ra tệp CSV mới tại:\n{new_path}")
            self.update_status(f"Đã xuất file CSV mới: {os.path.basename(new_path)}")
        except Exception as e:
            messagebox.showerror("Lỗi xuất file", f"Không thể lưu file CSV mới:\n{e}")


def main():
    root = tk.Tk()
    
    # Thiết lập icon / giao diện Tkinter
    try:
        root.tk.call('tk', 'scaling', 1.25) # Scale cho màn hình DPI cao
    except Exception:
        pass

    app = RelabelApp(root, CSV_FILE, IMAGE_DIR)
    root.mainloop()


if __name__ == "__main__":
    main()
