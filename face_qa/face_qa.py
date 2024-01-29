import mediapipe as mp
import cv2
import numpy as np

class FaceQA():
    '''
    image_path: Image file path (.jpg images)
    version: Face Classificator 1: haarcascade 2: mediapipe BlazeFace
    '''
    def __init__(self, image_path: str, version: int):
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

    def check_face(self):
        if self.version == 1:
            self.result['face_detected'], self.result['more_than_one_face'] = self._face_detection_v1()
            if self.result['face_detected'] == False:
                return self._return_all_false_result()
        elif self.version == 2:
            self.result['face_detected'], self.result['more_than_one_face'] = self._face_detection_v2()
            if self.result['face_detected'] == False:
                return self._return_all_false_result()
        
        # Haarcascade is default to validade other params
        model_path = 'models/haarcascade_frontalface_default.xml'
        image = cv2.imread(self.image_path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(model_path)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        self.result['eyes_is_good'] = self._eye_is_good(gray, image)

        self.result['is_smiling'] = self._is_smiling(face_cascade, image)
        
        self.result['contrast_is_good'] = self._contrast_is_good(gray)

        self.result['brightness_is_good'] = self._brightness_is_good(gray)

        self.result['face_is_centralized'] = self._face_is_centralized(image, faces)

        return self.result
            
    def _face_detection_v1(self) -> (bool, bool):
        '''
        Using HaarCascade to Face Classification
        '''
        model_path = 'models/haarcascade_frontalface_default.xml'
        image = cv2.imread(self.image_path)
        
        # To gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load model
        face_cascade = cv2.CascadeClassifier(model_path)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Check more than one face
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
        '''
        Using MediaPipe to Face Classification
        '''
        model_path = 'models/blaze_face_short_range.tflite'
        
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face detector instance with the image mode
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE)
        with FaceDetector.create_from_options(options) as detector:
            
            # Load the input image from an image file.
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
        '''
        Using np to verify
        '''
        x, y, w, h = faces[0]

        # Check if the face image is centered
        face_center = (x + w // 2, y + h // 2)
        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        distance = np.sqrt((face_center[0] - image_center[0]) ** 2 + (face_center[1] - image_center[1]) ** 2)

        if distance > image.shape[0] / 4:
            return False
        else:
            return True

    def _brightness_is_good(self, face) -> bool:
        '''
        Using cv2 to verify
        '''
        threshold = 120 # threshold

        # Make sure your face is clearly lit
        if cv2.mean(face)[0] < threshold:
            return False
        else:
            return True

    def _contrast_is_good(self, face) -> bool:
        '''
        Using cv2 to verify
        '''
        threshold = 70 # threshold

        # Make sure your iamge is clearly lit
        if cv2.meanStdDev(face)[1][0] < threshold:
            return False
        else:
            return True


    def _eye_is_good(self, gray_image, face_image) -> bool:
        '''
        Using HaarCascade to Eyes Classification
        '''
        model_path = 'models/haarcascade_eye.xml'

        eye_cascade = cv2.CascadeClassifier(model_path)
        
        eyes = eye_cascade.detectMultiScale(face_image, minNeighbors = 4)
        
        # Make sure there is at least one open eye in the image
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_image,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
            cv2.putText(face_image, "Olhos", (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            eye_roi = gray_image[ey:ey+eh, ex:ex+ew]
            threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)[1]
            cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 400: # threshold
                    return True
        return False

    def _is_smiling(self, face_cascade, face_image) -> bool:
        '''
        Using HaarCascade to Face Classification
        '''
        model_path = 'models/haarcascade_smile.xml'

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        smile_cascade = cv2.CascadeClassifier(model_path)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:     
            roi_gray = gray[y:y+h, x:x+w]
            
            # detecting smile within the face roi
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 40)
            if len(smiles) > 0:
                return True
            else:
                return False
        return False

    def _return_all_false_result(self):
        self.result['face_detected'] = False
        self.result['more_than_one_face']
        self.result['eyes_is_good'] = False
        self.result['is_smiling'] = False
        self.result['contrast_is_good'] = False
        self.result['brightness_is_good'] =  False
        self.result['face_is_centralized'] = False
        return self.result