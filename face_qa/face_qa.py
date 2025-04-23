import mediapipe as mp
import cv2
import numpy as np
import os
import json

class FaceQA():
    def __init__(self, image_path: str, version: int, config_path: str = '/config.json'):
        self.image_path = image_path
        self.version = version
        self.result = {
            "face_detected": bool,
            "more_than_one_face": bool,
            "eyes_is_good": bool,
            "is_smiling": bool,
            "contrast_is_good": bool,
            "brightness_is_good": bool,
            "face_is_centralized": bool
        }
        
        # Carregar configurações a partir do arquivo JSON
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Carrega o arquivo JSON de configuração."""
        path_file = os.path.dirname(__file__) + config_path
        # print(f"Loading config from: {path_file}")
        with open(path_file, 'r') as config_file:
            config = json.load(config_file)
        return config

    def check_face(self):
        if self.version == 1:
            self.result['face_detected'], self.result['more_than_one_face'] = self._face_detection_v1()
            if not self.result['face_detected']:
                return self._return_all_false_result()
        elif self.version == 2:
            self.result['face_detected'], self.result['more_than_one_face'] = self._face_detection_v2()
            if not self.result['face_detected']:
                return self._return_all_false_result()
        
        # Haarcascade para validação dos outros parâmetros
        model_path = os.path.dirname(__file__) + '/models/haarcascade_frontalface_default.xml'
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(model_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=self.config["scale_factor_face_cascade"], minNeighbors=self.config["min_neighbors_face_cascade"], minSize=self.config["min_size_face_cascade"])

        # Usar as configurações carregadas do JSON
        self.result['eyes_is_good'] = self._eye_is_good(gray, image)
        self.result['is_smiling'] = self._is_smiling(face_cascade, image)
        self.result['contrast_is_good'] = self._contrast_is_good(gray)
        self.result['brightness_is_good'] = self._brightness_is_good(gray)
        self.result['face_is_centralized'] = self._face_is_centralized(image, faces)

        return self.result

    def _face_detection_v1(self) -> (bool, bool):
        model_path = os.path.dirname(__file__) + '/models/haarcascade_frontalface_default.xml'
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(model_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:         
            face_detection = True
            more_than_one_face = False
        elif len(faces) > 1:
            face_detection = True
            more_than_one_face = True
        else:
            face_detection = False
            more_than_one_face = False

        return face_detection, more_than_one_face
    
    def _face_detection_v2(self) -> bool:
        model_path = os.path.dirname(__file__) + '/models/blaze_face_short_range.tflite'
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
        with FaceDetector.create_from_options(options) as detector:
            mp_image = mp.Image.create_from_file(self.image_path)
            face_detector_result = detector.detect(mp_image)
            if face_detector_result.detections:
                face_detection = True
                if len(face_detector_result.detections) > 1:
                    more_than_one_face = True
                else:
                    more_than_one_face = False
            else:
                face_detection = False
                more_than_one_face = False
                
            return face_detection, more_than_one_face

    def _face_is_centralized(self, image, faces) -> bool:
        x, y, w, h = faces[0]
        face_center = (x + w // 2, y + h // 2)
        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        distance = np.sqrt((face_center[0] - image_center[0]) ** 2 + (face_center[1] - image_center[1]) ** 2)
        
        # Desenhar centro
        annotated = image.copy()
        cv2.circle(annotated, face_center, 5, (0, 255, 0), -1)
        cv2.circle(annotated, image_center, 5, (255, 0, 0), -1)
        self._save_image_result_folder_output(annotated, prefix="center_")

        return distance <= image.shape[0] * self.config["face_center_threshold"]

    def _brightness_is_good(self, gray) -> bool:
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        brightness = cv2.mean(gray)[0]
        cv2.putText(annotated, f"Brightness: {brightness:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
        self._save_image_result_folder_output(annotated, prefix="brightness_")
        return brightness >= self.config["brightness_threshold"]

    def _contrast_is_good(self, gray) -> bool:
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        contrast = cv2.meanStdDev(gray)[1][0][0]  # Extração correta do valor escalar
        cv2.putText(annotated, f"Contrast: {contrast:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 6)
        self._save_image_result_folder_output(annotated, prefix="contrast_")
        return contrast >= self.config["contrast_threshold"]

    def _eye_is_good(self, gray_image, face_image) -> bool:
        model_path = os.path.dirname(__file__) + '/models/haarcascade_eye.xml'
        eye_cascade = cv2.CascadeClassifier(model_path)
        eyes = eye_cascade.detectMultiScale(face_image, minNeighbors=4)
        annotated = face_image.copy()

        # get maiores olhos
        eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]
        
        good_eye = False
        for (ex, ey, ew, eh) in eyes:
            eye_roi = gray_image[ey:ey+eh, ex:ex+ew]
            thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for c in cnts:
                area = cv2.contourArea(c)
                if area > self.config["eye_area_threshold"]:
                    good_eye = True
            cv2.rectangle(annotated, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        self._save_image_result_folder_output(annotated, prefix="eyes_")
        return good_eye

    def _is_smiling(self, face_cascade, face_image) -> bool:
        model_path = os.path.dirname(__file__) + '/models/haarcascade_smile.xml'
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        smile_cascade = cv2.CascadeClassifier(model_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        annotated = face_image.copy()

        smiling = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cascade.detectMultiScale(roi_gray, 2.0, self.config["smile_area_threshold"])
            if len(smiles) > 0:
                smiling = True
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(annotated, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 255), 2)

        self._save_image_result_folder_output(annotated, prefix="smile_")
        return smiling

    def _return_all_false_result(self):
        self.result['face_detected'] = False
        self.result['more_than_one_face'] = False
        self.result['eyes_is_good'] = False
        self.result['is_smiling'] = False
        self.result['contrast_is_good'] = False
        self.result['brightness_is_good'] = False
        self.result['face_is_centralized'] = False
        return self.result
    
    def _save_image_result_folder_output(self, image, folder_path = 'output', prefix = 'result_'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        output_path = os.path.join(folder_path, f"{prefix}.png")
        cv2.imwrite(output_path, image)
        # print(f"Image saved to {output_path}")