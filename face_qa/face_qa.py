import mediapipe as mp
import cv2
import numpy as np
import os
import json

class FaceQA():
    '''
    image_path: Image file path (.jpg images)
    version: Face Classificator 1: haarcascade 2: mediapipe BlazeFace
    '''
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
        
        # Haarcascade is default to validade other params
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

        # Corta apenas a parte superior do rosto
        height = face_image.shape[0] + self.config["face_height_adcional"]
        upper_face_image = face_image[:height // 2, :]
        upper_gray_image = gray_image[:height // 2, :]

        self._save_image_result_folder_output(upper_gray_image, prefix="heigth_eyes_")

        # Detecta olhos só na parte de cima
        eyes = eye_cascade.detectMultiScale(upper_face_image, minNeighbors=4)
        annotated = upper_face_image.copy()

        # Pega os dois maiores olhos
        eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)[:2]

        good_eye = False
        for (ex, ey, ew, eh) in eyes:
            # Corrige ey para coordenada da imagem original
            absolute_ey = ey  # já é parte superior, então não precisa somar offset

            eye_roi = upper_gray_image[absolute_ey:absolute_ey+eh, ex:ex+ew]
            thresh = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for c in cnts:
                area = cv2.contourArea(c)
                if area > self.config["eye_area_threshold"]:
                    good_eye = True

            # Corrige coordenadas para desenhar na imagem completa
            cv2.rectangle(annotated, (ex, absolute_ey), (ex+ew, absolute_ey+eh), (0, 255, 0), 2)

        self._save_image_result_folder_output(annotated, prefix="eyes_")
        return good_eye

    def _is_smiling(self, _, face_image) -> bool:
        import mediapipe as mp
        import numpy as np
        import cv2

        mp_face_mesh = mp.solutions.face_mesh
        annotated = face_image.copy()
        smile_ratio_threshold = self.config.get("smile_ratio_threshold", 1.8)
        min_mouth_width = self.config.get("min_mouth_width", 40)

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                self._save_image_result_folder_output(annotated, prefix="smile_")
                return False

            landmarks = results.multi_face_landmarks[0].landmark
            img_h, img_w, _ = face_image.shape

            def to_pixel(landmark):
                return np.array([landmark.x * img_w, landmark.y * img_h])

            # Posição dos pontos da boca
            left_pt = to_pixel(landmarks[61])
            right_pt = to_pixel(landmarks[291])
            top_pt = to_pixel(landmarks[13])
            bottom_pt = to_pixel(landmarks[14])

            mouth_width = np.linalg.norm(right_pt - left_pt)
            mouth_height = np.linalg.norm(top_pt - bottom_pt)
            smile_ratio = mouth_width / (mouth_height + 1e-6)



            # Validações extras
            if mouth_width < min_mouth_width:
                smiling = False
            elif smile_ratio < 1.1:
                smiling = False
            else:
                smiling = smile_ratio < smile_ratio_threshold

            # Desenho e debug
            cv2.putText(annotated, f"Smile Ratio: {smile_ratio:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
            cv2.line(annotated, tuple(left_pt.astype(int)), tuple(right_pt.astype(int)), (0, 255, 255), 2)
            cv2.line(annotated, tuple(top_pt.astype(int)), tuple(bottom_pt.astype(int)), (255, 0, 255), 2)

            self._save_image_result_folder_output(annotated, prefix="smile_")
            return smiling

    def _return_all_false_result(self):
        self.result['face_detected'] = False
        self.result['more_than_one_face'] = False
        self.result['eyes_is_good'] = False
        self.result['is_smiling'] = False
        self.result['contrast_is_good'] = False
        self.result['brightness_is_good'] =  False
        self.result['face_is_centralized'] = False
        return self.result
    
    def _save_image_result_folder_output(self, image, folder_path = 'output', prefix = 'result_'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        output_path = os.path.join(folder_path, f"{prefix}.png")
        cv2.imwrite(output_path, image)