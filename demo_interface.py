from tkinter import *
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from face_qa.face_qa import FaceQA

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

class InterfaceDemo(customtkinter.CTk):

    WIDTH = 600
    HEIGHT = 600

    def __init__(self):
        super().__init__()

        self.title("InterfaceDemo")
        self.geometry(f"{InterfaceDemo.WIDTH}x{InterfaceDemo.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed


        self.frame_menu = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_menu.pack()

        self.frame_initial = customtkinter.CTkFrame(master=self)
        self.frame_initial.pack()

        # ============ frame_menu ============
        self.frame_menu.pack()

        self.label_1 = customtkinter.CTkLabel(master=self.frame_menu,
                                              text="InterfaceDemo",
                                              ) 
        self.label_1.pack()

        # ============.frame_initial ============
        self.frame_initial.rowconfigure((0, 1, 2, 3), weight=1)
        self.frame_initial.rowconfigure(7, weight=10)
        self.frame_initial.columnconfigure((0, 1), weight=1)
        self.frame_initial.columnconfigure(2, weight=0)

        # ============ frame_info =========
        self.frame_info = customtkinter.CTkFrame(master=self)
        self.frame_info.pack(side=TOP)

        self.button_1 = customtkinter.CTkButton(master=self.frame_info,
                                                text="Select a Image",
                                                command=self.load_image
                                                )
        self.button_1.pack()

        self.image = None 

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def load_image(self):
        file_path = filedialog.askopenfilename(title='Select a image')
        image = Image.open(file_path)

        self.image_check(file_path)

        # Remove image
        if self.image:
            self.image_label.configure(image=None)
            self.image_label.image = None

        # Resize image
        if image.width > 200 or image.height > 200:
            ratio = min(200/image.width, 200/image.height)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.Resampling.LANCZOS)

        if hasattr(self, "image_label"):
            self.image_label.pack_forget()
            self.image_label.destroy()

        photo = ImageTk.PhotoImage(image)
        self.image_label = Label(self.frame_initial, image=photo)
        self.image_label.image = photo
        self.image_label.pack()

        self.image = image

    def image_check(self, file_path):
        validator = FaceQA(file_path, 2)
        result = validator.check_face()
        
        if hasattr(self, "label_face"):
            self.label_face.pack_forget()
            self.label_face.destroy()
        if hasattr(self, "label_eyes"):
            self.label_eyes.pack_forget()
            self.label_eyes.destroy()
        if hasattr(self, "is_smiling"):
            self.is_smiling.pack_forget()
            self.is_smiling.destroy()
        if hasattr(self, "contrast_is_good"):
            self.contrast_is_good.pack_forget()
            self.contrast_is_good.destroy()
        if hasattr(self, "brightness_is_good"):
            self.brightness_is_good.pack_forget()
            self.brightness_is_good.destroy()
        if hasattr(self, "face_is_centralized"):
            self.face_is_centralized.pack_forget()
            self.face_is_centralized.destroy()
        if hasattr(self, "more_than_one_face"):
            self.more_than_one_face.pack_forget()
            self.more_than_one_face.destroy()
        
        if not result["face_detected"]:
            self.label_face = customtkinter.CTkLabel(master=self.frame_info,
                                                    text="Face was not detected correctly.",
                                                    font=("Helvetica", 22),
                                                    text_color="red")
            self.label_face.pack()
        else:
            self.label_face = customtkinter.CTkLabel(master=self.frame_info,
                                                    text="Face detected.",
                                                    font=("Helvetica", 22),
                                                    text_color="green")
            self.label_face.pack()
            if result["more_than_one_face"]:
                self.more_than_one_face = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="More than one face in image",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.more_than_one_face.pack()
            else:
                self.more_than_one_face = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Only one face detected",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.more_than_one_face.pack()
            if result["eyes_is_good"]:
                self.label_eyes = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Eyes are clear and visible",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.label_eyes.pack()
            else:
                self.label_eyes = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Eyes are closed or not visible",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.label_eyes.pack()
            if result["is_smiling"]:
                self.is_smiling = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Smile detected",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.is_smiling.pack()
            else:
                self.is_smiling = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Smile not detected",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.is_smiling.pack()
            if result["contrast_is_good"]:
                self.contrast_is_good = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Image contrast is good",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.contrast_is_good.pack()
            else:
                self.contrast_is_good = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Image contrast isn't good",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.contrast_is_good.pack()
            if result["brightness_is_good"]:
                self.brightness_is_good = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Image brightness is good",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.brightness_is_good.pack()
            else:
                self.brightness_is_good = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Image brightness isn't good",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.brightness_is_good.pack()
            if result["face_is_centralized"]:
                self.face_is_centralized = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Face is centralized",
                                                        font=("Helvetica", 22),
                                                        text_color="green")
                self.face_is_centralized.pack()
            else:
                self.face_is_centralized = customtkinter.CTkLabel(master=self.frame_info,
                                                        text="Face isn't centralized",
                                                        font=("Helvetica", 22),
                                                        text_color="red")
                self.face_is_centralized.pack()

    def on_closing(self, event=0):
        self.destroy()

if __name__ == "__main__":
    app = InterfaceDemo()
    app.mainloop()
