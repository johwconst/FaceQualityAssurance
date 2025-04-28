import json
from tkinter import *
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from face_qa.face_qa import FaceQA
import os
import re

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")


class FaceQAViewer(customtkinter.CTk):

    WIDTH = 800
    HEIGHT = 800

    def __init__(self):
        super().__init__()

        self.title("FaceQA Viewer")
        self.geometry(f"{FaceQAViewer.WIDTH}x{FaceQAViewer.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.image_paths = []
        self.current_index = 0

        self.thresholds = self.load_config()
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
            text="Select folder",
            command=self.select_folder
        )
        self.button_select_folder.pack(pady=5)

        self.button_next = customtkinter.CTkButton(
            master=self.frame_info,
            text="Next image",
            command=self.next_image
        )
        self.button_next.pack(pady=5)

        self.button_previous = customtkinter.CTkButton(
            master=self.frame_info,
            text="Previous image",
            command=self.previous_image
        )
        self.button_previous.pack(pady=5)

        # Sliders
        self.add_slider("scale_factor_face_cascade", "Face scale factor", 1, 10, 0.1)
        self.add_slider("min_neighbors_face_cascade", "Min neighbors for face", 1, 50)
        self.add_slider("brightness_threshold", "Min brightness", 0, 255)
        self.add_slider("contrast_threshold", "Min contrast", 0, 100)
        self.add_slider("face_center_threshold", "Center deviation", 0, 1, 0.01)
        self.add_slider("face_height_adcional", "Extra face height", 0, 1000, 1)
        self.add_slider("eye_area_threshold", "Eye area", 0, 1, 0.01)
        self.add_slider("smile_ratio_threshold", "Smile area", 0, 100, 1)

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'face_qa/config.json')
            with open(file_path, 'r') as file:
                config = json.load(file)
            return config
        except Exception as e:
            print(f"Error loading config file: {e}")
            exit(1)

    def save_config(self):
        """Save configuration to JSON file"""
        try:
            file_path = os.path.join(os.path.dirname(__file__), 'face_qa/config.json')
            with open(file_path, 'w') as file:
                json.dump(self.thresholds, file, indent=4)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def add_slider(self, key, label_text, min_val, max_val, step=1):
        def update_slider(value):
            self.thresholds[key] = float(value) if isinstance(step, float) else int(value)
            self.save_config()
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
        folder_path = filedialog.askdirectory(title='Select image folder')
        print(f"Selected folder: {folder_path}")
        if not folder_path:
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]
        self.image_paths.sort(key=lambda x: int(re.search(r'\d+', os.path.splitext(os.path.basename(x))[0]).group()))
        self.current_index = 0
        if self.image_paths:
            self.load_image(self.image_paths[self.current_index])

    def next_image(self):
        if self.current_index + 1 < len(self.image_paths):
            self.current_index += 1
            self.load_image(self.image_paths[self.current_index])
        else:
            print("End of images.")

    def previous_image(self):
        if self.current_index - 1 >= 0:
            self.current_index -= 1
            self.load_image(self.image_paths[self.current_index])
        else:
            print("Beginning of images.")

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
            print(f"Error loading image {file_path}: {e}")

    def image_check(self, file_path):
        validator = FaceQA(image_path=file_path, version=2)
        result = validator.check_face()

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

        if not result["face_detected"]:
            create_label("label_face", "Face not detected.", False)
        else:
            create_label("label_face", "Face detected.", True)
            create_label("more_than_one_face", "More than one face" if result["more_than_one_face"] else "Only one face", not result["more_than_one_face"])
            create_label("label_eyes", "Eyes visible" if result["eyes_is_good"] else "Eyes not visible", result["eyes_is_good"])
            create_label("is_smiling", "Smiling" if result["is_smiling"] else "No smile", not result["is_smiling"])
            create_label("contrast_is_good", "Good contrast" if result["contrast_is_good"] else "Poor contrast", result["contrast_is_good"])
            create_label("brightness_is_good", "Good brightness" if result["brightness_is_good"] else "Poor brightness", result["brightness_is_good"])
            create_label("face_is_centralized", "Face centered" if result["face_is_centralized"] else "Face not centered", result["face_is_centralized"])

        for widget in self.frame_initial.winfo_children():
            if isinstance(widget, Label) and widget != self.image_label:
                widget.destroy()

        output_dir = os.path.join("output")
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith((".png", ".jpg")):
                    try:
                        img_path = os.path.join(output_dir, file)
                        img = Image.open(img_path)
                        img = img.resize((150, 150), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        label = Label(self.frame_initial, image=photo)
                        label.image = photo
                        label.pack(side=RIGHT, padx=5)
                    except Exception as e:
                        print(f"Error displaying image {file}: {e}")

    def on_closing(self, event=0):
        self.destroy()


if __name__ == "__main__":
    app = FaceQAViewer()
    app.mainloop()
