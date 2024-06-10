# app.py

import tkinter as tk
from tkinter import filedialog, Text
from PIL import ImageTk, Image
import backend

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        
        self.create_widgets()

    def create_widgets(self):
        # Khu vực hiển thị ảnh upload
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.grid(row=0, column=0, rowspan=3)

        # Button để upload ảnh
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=1)

        # Button để thực hiện các thao tác xử lý ảnh
        self.process_button1 = tk.Button(self.root, text="Enhance", command=lambda: self.process_image("enhance"))
        self.process_button1.grid(row=1, column=1)

        self.process_button2 = tk.Button(self.root, text="Gray", command=lambda: self.process_image("gray"))
        self.process_button2.grid(row=2, column=1)

        # Hộp văn bản hiển thị kết quả
        self.result_text = Text(self.root, height=10, width=30)
        self.result_text.grid(row=3, column=0, columnspan=2)

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.original_image = backend.load_image(self.file_path)
            self.display_image(self.original_image)
            self.result_text.insert(tk.END, f"Image {self.file_path} uploaded successfully.\n")

    def process_image(self, action):
        if hasattr(self, 'original_image'):
            self.processed_image = backend.process_image(self.original_image, action)
            self.display_image(self.processed_image)
            self.result_text.insert(tk.END, f"Image processed with action: {action}.\n")

    def display_image(self, image):
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
