import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import material_price_estimator as analyzer

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisi Prodotti Handmade")
        self.root.geometry("700x700")

        self.image_path = None

        # Pulsante carica immagine
        self.load_btn = tk.Button(root, text="Carica Foto", command=self.load_image)
        self.load_btn.pack(pady=10)

        # Campo larghezza reale
        self.width_label = tk.Label(root, text="Larghezza reale (cm):")
        self.width_label.pack()
        self.width_entry = tk.Entry(root)
        self.width_entry.pack()

        # Pulsante analizza
        self.analyze_btn = tk.Button(root, text="Analizza", command=self.analyze)
        self.analyze_btn.pack(pady=10)

        # Area risultati
        self.result_text = tk.Text(root, height=15)
        self.result_text.pack(pady=10)

        # Area immagine
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.image_path = path
            img = Image.open(path)
            img.thumbnail((400, 400))
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image)

    def analyze(self):
        if not self.image_path:
            messagebox.showerror("Errore", "Carica prima una foto.")
            return

        try:
            real_width = float(self.width_entry.get())
        except:
            messagebox.showerror("Errore", "Inserisci una larghezza valida.")
            return

        self.result_text.delete("1.0", tk.END)

        result = analyzer.analyze_image(
            self.image_path,
            real_width_cm=real_width
        )

        materials = result["material_scores"]
        prices = result["price_info"]["price_suggestion"]
        costs = result["price_info"]["costs"]

        self.result_text.insert(tk.END, "Materiali stimati:\n")
        for m, s in list(materials.items())[:5]:
            self.result_text.insert(tk.END, f"- {m}: {round(s*100,1)}%\n")

        self.result_text.insert(tk.END, "\nCosti:\n")
        self.result_text.insert(tk.END, f"Materiale: €{round(costs['material_cost'],2)}\n")
        self.result_text.insert(tk.END, f"Lavoro: €{round(costs['labor_cost'],2)}\n")
        self.result_text.insert(tk.END, f"Base: €{round(costs['base_cost'],2)}\n")

        self.result_text.insert(tk.END, "\nPrezzo suggerito:\n")
        self.result_text.insert(tk.END, f"Basso: €{round(prices['low'],2)}\n")
        self.result_text.insert(tk.END, f"Tipico: €{round(prices['typical'],2)}\n")
        self.result_text.insert(tk.END, f"Alto: €{round(prices['high'],2)}\n")


root = tk.Tk()
app = App(root)
root.mainloop()
