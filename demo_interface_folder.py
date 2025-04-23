import json
from tkinter import *
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from face_qa.face_qa import FaceQA
import os

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")


class InterfaceDemo(customtkinter.CTk):

    WIDTH = 800
    HEIGHT = 800

    def __init__(self):
        super().__init__()

        self.title("FaceQA Viewer")
        self.geometry(f"{InterfaceDemo.WIDTH}x{InterfaceDemo.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.image_paths = []
        self.current_index = 0

        self.thresholds = self.load_config()  # Carrega as configurações do JSON
        self.face_qa = None
        self.image = None

        # Frames
        self.frame_menu = customtkinter.CTkFrame(master=self, width=180, corner_radius=0)
        self.frame_menu.pack()

        self.frame_initial = customtkinter.CTkFrame(master=self)
        self.frame_initial.pack()

        self.frame_info = customtkinter.CTkFrame(master=self)
        self.frame_info.pack(side=LEFT, padx=10)

        self.frame_controls = customtkinter.CTkFrame(master=self)
        self.frame_controls.pack(side=RIGHT, padx=10)

        # Menu
        self.label_1 = customtkinter.CTkLabel(master=self.frame_menu, text="FaceQA Viewer")
        self.label_1.pack()

        self.button_select_folder = customtkinter.CTkButton(
            master=self.frame_info,
            text="Selecionar pasta",
            command=self.select_folder
        )
        self.button_select_folder.pack(pady=5)

        self.button_next = customtkinter.CTkButton(
            master=self.frame_info,
            text="Próxima imagem",
            command=self.next_image
        )
        self.button_next.pack(pady=5)

        self.button_previus = customtkinter.CTkButton(
            master=self.frame_info,
            text="Imagem anterior",
            command=self.previous_image
        )
        self.button_previus.pack(pady=5)

        # Sliders
        self.add_slider("brightness_threshold", "Brilho mínimo", 0, 255)
        self.add_slider("contrast_threshold", "Contraste mínimo", 0, 100)
        self.add_slider("face_center_threshold", "Desvio centro (0.0 - 1.0)", 0, 1, 0.01)
        self.add_slider("eye_area_threshold", "Área olhos (0.0 - 1.0)", 0, 1, 0.01)

        self.select_folder()

    def load_config(self):
        """Carrega as configurações do arquivo JSON"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'face_qa/config.json')
            # print(f"Loading config from: {file_path}")
            with open(file_path, 'r') as file:
                config = json.load(file)
            return config
        except Exception as e:
            print(f"Erro ao carregar o arquivo de configuração: {e}")
            return {
                "brightness_threshold": 100,
                "contrast": 30,
                "center": 0.35,
                "eye_area": 0.25,
            }

    def save_config(self):
        """Salva as configurações no arquivo JSON"""
        try:
            with open('config.json', 'w') as file:
                json.dump(self.thresholds, file, indent=4)
        except Exception as e:
            print(f"Erro ao salvar o arquivo de configuração: {e}")

    def add_slider(self, key, label_text, min_val, max_val, step=1):
        def update_slider(value):
            self.thresholds[key] = float(value)
            self.save_config()  # Salva o novo valor no arquivo JSON
            if self.image_paths:
                self.load_image(self.image_paths[self.current_index])

        label = customtkinter.CTkLabel(master=self.frame_controls, text=label_text)
        label.pack()
        slider = customtkinter.CTkSlider(master=self.frame_controls,
                                         from_=min_val, to=max_val,
                                         number_of_steps=int((max_val - min_val) / step),
                                         command=update_slider)
        slider.set(self.thresholds[key])
        slider.pack(pady=5)

    def select_folder(self):
        # folder_path = filedialog.askdirectory(title='Selecionar pasta de imagens')
        folder_path = "/home/augusto.savi/Documentos/github/FaceQualityAssurance/fotos"
        print(f"Selecionando pasta: {folder_path}")
        if not folder_path:
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]

        self.image_paths.sort()
        self.current_index = 0
        if self.image_paths:
            self.load_image(self.image_paths[self.current_index])

    def next_image(self):
        if self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self.load_image(self.image_paths[self.current_index])
        else:
            print("Fim das imagens.")
    
    def previous_image(self):
        if self.current_index - 1 >= 0:
            self.current_index -= 1
            self.load_image(self.image_paths[self.current_index])
        else:
            print("Início das imagens.")

    def load_image(self, file_path):
        self.label_1.configure(text=os.path.basename(file_path))

        try:
            image = Image.open(file_path)
            self.image_check(file_path)

            if self.image:
                self.image_label.configure(image=None)
                self.image_label.image = None

            if image.width > 300 or image.height > 300:
                ratio = min(300 / image.width, 300 / image.height)
                image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)

            if hasattr(self, "image_label"):
                self.image_label.pack_forget()
                self.image_label.destroy()

            photo = ImageTk.PhotoImage(image)
            self.image_label = Label(self.frame_initial, image=photo)
            self.image_label.image = photo
            self.image_label.pack()

            self.image = image
        except Exception as e:
            print(f"Erro ao carregar imagem {file_path}: {e}")

    def image_check(self, file_path):
        validator = FaceQA(
            image_path=file_path,
            version=2
        )
        result = validator.check_face()

        # Limpa labels anteriores
        for attr in ["label_face", "label_eyes", "is_smiling", "contrast_is_good",
                     "brightness_is_good", "face_is_centralized", "more_than_one_face"]:
            if hasattr(self, attr):
                getattr(self, attr).pack_forget()
                getattr(self, attr).destroy()

        def create_label(attr_name, text, ok, color_ok="green", color_bad="red"):
            label = customtkinter.CTkLabel(
                master=self.frame_info,
                text=text,
                font=("Helvetica", 18),
                text_color=color_ok if ok else color_bad
            )
            setattr(self, attr_name, label)
            label.pack()

        # Resultados
        if not result["face_detected"]:
            create_label("label_face", "Face não detectada.", False)
        else:
            create_label("label_face", "Face detectada.", True)
            create_label("more_than_one_face", "Mais de uma face" if result["more_than_one_face"] else "Apenas uma face", not result["more_than_one_face"])
            create_label("label_eyes", "Olhos visíveis" if result["eyes_is_good"] else "Olhos não visíveis", result["eyes_is_good"])
            create_label("is_smiling", "Sorriso detectado" if result["is_smiling"] else "Sem sorriso", not result["is_smiling"])
            create_label("contrast_is_good", "Contraste bom" if result["contrast_is_good"] else "Contraste ruim", result["contrast_is_good"])
            create_label("brightness_is_good", "Brilho bom" if result["brightness_is_good"] else "Brilho ruim", result["brightness_is_good"])
            create_label("face_is_centralized", "Face centralizada" if result["face_is_centralized"] else "Face não centralizada", result["face_is_centralized"])

        # limpar anteriores
        for widget in self.frame_initial.winfo_children():
            if isinstance(widget, Label) and widget != self.image_label:
                widget.destroy()

        # Mostrar imagens auxiliares
        output_dir = os.path.join("output")
        # print(f"Verificando imagens auxiliares em: {output_dir}")
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith((".png", ".jpg")):
                    try:
                        # print(f"Carregando imagem auxiliar: {file}")
                        img_path = os.path.join(output_dir, file)
                        img = Image.open(img_path)
                        img = img.resize((150, 150), Image.ANTIALIAS)
                        photo = ImageTk.PhotoImage(img)
                        label = Label(self.frame_initial, image=photo)
                        label.image = photo
                        label.pack(side=RIGHT, padx=5)
                    except Exception as e:
                        print(f"Erro ao mostrar imagem {file}: {e}")

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = InterfaceDemo()
    app.mainloop()
