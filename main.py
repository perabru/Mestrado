import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Listbox, Scrollbar, Label, Entry, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

class AnalisadorIonosferico(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analisador de Ionosfera")
        self.geometry("800x600")

        # Campo para inserir a escala de pixel por metro
        self.scale_frame = tk.Frame(self)
        self.scale_frame.pack()
        self.scale_label = Label(self.scale_frame, text="Digite a escala de pixel por metro:")
        self.scale_label.pack(side=tk.LEFT)
        self.scale_entry = Entry(self.scale_frame)
        self.scale_entry.pack(side=tk.LEFT)
        
        # Botão para carregar imagem
        self.load_button = tk.Button(self, text="Carregar Imagem", command=self.carregar_imagem)
        self.load_button.pack()

        # Botão para processar imagem
        self.process_button = tk.Button(self, text="Processar Imagem", command=self.processar_imagem)
        self.process_button.pack()

        # Canvas para mostrar a imagem
        self.canvas = tk.Canvas(self, width=800, height=500, bg='gray')
        self.canvas.pack()

        # Imagem original e processada
        self.original_image = None
        self.processed_image = None
        self.image_path = None

    def carregar_imagem(self):
        path = filedialog.askopenfilename()
        if path:
            self.image_path = path
            self.original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is not None:
                print(f"Imagem carregada com sucesso de: {path}")
                self.mostrar_imagem(self.original_image)
            else:
                messagebox.showerror("Erro", "Falha ao carregar imagem. Verifique o arquivo.")

    def mostrar_imagem(self, image):
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        self.canvas.image = photo
        self.canvas.create_image(0, 0, image=self.canvas.image, anchor='nw')

    def processar_imagem(self):
        if self.original_image is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada.")
            return

        image_processed = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        _, bin_image = cv2.threshold(image_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.processed_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        for i, contorno in enumerate(contornos):
            cv2.drawContours(self.processed_image, [contorno], -1, (0, 255, 0), 2)
            moments = cv2.moments(contorno)
            if moments['m00'] != 0:
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                cv2.putText(self.processed_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        self.mostrar_imagem(self.processed_image)
        self.mostrar_resultados(contornos)

    def mostrar_resultados(self, contornos):
        results_window = Toplevel(self)
        results_window.title("Resultados da Análise de Contornos")
        results_window.geometry("300x400")
        scroll_bar = Scrollbar(results_window)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        results_list = Listbox(results_window, yscrollcommand=scroll_bar.set)
        pixel_to_meter = self.get_pixel_to_meter()
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)
            perimetro = cv2.arcLength(contorno, True)
            if pixel_to_meter:
                area *= (pixel_to_meter ** 2)
                perimetro *= pixel_to_meter
                results_list.insert(tk.END, f"Contorno {i+1}: Área = {area:.4f} m², Perímetro = {perimetro:.4f} m")
            else:
                results_list.insert(tk.END, f"Contorno {i+1}: Área = {area:.2f} px², Perímetro = {perimetro:.2f} px")
        results_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_bar.config(command=results_list.yview)
        pdf_button = Button(results_window, text="Gerar Relatório PDF", command=lambda: self.create_pdf(contornos, pixel_to_meter))
        pdf_button.pack()

    def get_pixel_to_meter(self):
        try:
            pixel_to_meter = float(self.scale_entry.get())
            if pixel_to_meter <= 0:
                raise ValueError("A escala deve ser positiva e maior que zero.")
            return pixel_to_meter
        except ValueError:
            return None

    def create_pdf(self, contornos, pixel_to_meter):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        
        c = canvas.Canvas(file_path, pagesize=letter)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawImage(ImageReader(Image.fromarray(self.processed_image)), 50, 500, width=500, height=300)  # Place the image at the top
        text = c.beginText(50, 480)
        text.setFont("Helvetica", 12)
        text.textLine("Resultados da Análise de Contornos:")
        if pixel_to_meter:
            for i, contorno in enumerate(contornos):
                area = cv2.contourArea(contorno) * (pixel_to_meter ** 2)
                perimetro = cv2.arcLength(contorno, True) * pixel_to_meter
                text.textLine(f"Contorno {i+1}: Área = {area:.4f} m², Perímetro = {perimetro:.4f} m")
        else:
            for i, contorno in enumerate(contornos):
                area = cv2.contourArea(contorno)
                perimetro = cv2.arcLength(contorno, True)
                text.textLine(f"Contorno {i+1}: Área = {area:.2f} px², Perímetro = {perimetro:.2f} px")
        c.drawText(text)
        # Place the timestamp at the bottom of the page
        c.drawString(50, 30, f"Generated on: {timestamp}")
        c.showPage()
        c.save()
        messagebox.showinfo("PDF Gerado", "O relatório PDF foi gerado com sucesso.")

if __name__ == "__main__":
    app = AnalisadorIonosferico()
    app.mainloop()

