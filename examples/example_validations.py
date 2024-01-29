from face_qa.face_qa import FaceQA

# IMPORTANT!! move to root folder to run 

image_path = 'images/image_2.jpg'

face_qa = FaceQA(image_path, 1)

result = face_qa.check_face()

print('Target: ', str(image_path))
print('############# RESULT #############')
print("face_detected", result['face_detected'])
print("more_than_one_face", result['more_than_one_face'])
print("eyes_is_good", result['eyes_is_good'])
print("is_smiling", result['is_smiling'])
print("contrast_is_good", result['contrast_is_good'])
print("brightness_is_good", result['brightness_is_good'])
print("face_is_centralized", result['face_is_centralized'] )
print('#####################################')
