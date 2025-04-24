<p class="header" align="center">
 <img width="100px" src="https://i.imgur.com/DqtESPx.png" align="center" alt="Image" />
 <h2 align="center">FaceQualityAssurance</h2>
 <p align="center">FaceQualityAssurance is a tool to validate the quality of faces images for registration on devices that use facial recognition</p>
</p>
<p align="center">
  <a href="https://github.com/johwconst/FaceQualityAssurance/graphs/contributors">
    <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/johwconst/FaceQualityAssurance?color=0088ff" />
  </a>
  <a href="https://github.com/johwconst/FaceQualityAssurance/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/johwconst/FaceQualityAssurance?color=0088ff" />
  </a>
  <a href="https://github.com/johwconst/FaceQualityAssurance/pulls">
    <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/johwconst/FaceQualityAssurance?color=0088ff" />
  </a>
  <br />
  <br />
</p>
<p class="links" align="center">
  <a href="#why-did-you-do-that">Why did you do that?</a>
  .
  <a href="#how-run">How Run</a>
  .
  <a href="https://github.com/johwconst/FaceQualityAssurance/issues/new?template=feature.yaml"">Request Improvement</a>
  .
  <a href="https://github.com/johwconst/FaceQualityAssurance/new?template=feature.yaml">Report Bugs</a>
  .
  <a href="https://github.com/johwconst/FaceQualityAssurance/blob/main/CONTRIBUTING.md">Contributing</a>
</p>
 <br />

# Why did you do that?
The effectiveness of face registration in systems or devices utilizing facial recognition is crucial to ensure accurate user recognition. Currently, the lack of straightforward tools for face validation in this context necessitates the manual compilation of desired parameters and requirements.

The aim of this project is to create a tool for validating the quality of these images before they are sent to the facial recognition device or system.

<p class="header" align="center">
 <img width="200px" src="https://i.imgur.com/YmtxbCM.png" align="center" alt="Image" />
 <img width="200px" src="https://i.imgur.com/R3Uf1XU.png" align="center" alt="Image" />
 <img width="200px" src="https://i.imgur.com/MGwJRWX.png" align="center" alt="Image" />
</p>

## Demo interface folder

![image](https://github.com/user-attachments/assets/94d25d35-ec96-448f-9694-e13f7170e810)
![image](https://github.com/user-attachments/assets/08c7b366-4c5b-4e55-b88b-b810a095ed99)
![image](https://github.com/user-attachments/assets/19f128ff-0dd6-4820-97c3-b09a5508f71b)
![image](https://github.com/user-attachments/assets/fe3ed821-14e8-4787-9d31-286c145e05df)


## Verified Features
- Verify if there is a face in the image ✔️
- Verify if there is more than one face in the image ✔️
- Check if the face is well-lit ✔️
- Check if the face is centered ✔️
- check whether the face is illuminated ✔️
- Ensure there is a neutral expression (Smile detection) ✔️
- Verify if the eyes are present and open ✔️

## Classificators Used

Check | Classificator |
-- |-- |
Face | HeerCascade or MediaPipe
Smile | MediaPipe 
Eyes | HeerCascade

## Interface Demo
<p>
<a href="https://github.com/johwconst/FaceQualityAssurance/releases/download/1.0.0/demo_interface.exe"">Download .EXE Interface Demo</a>
</p>

## Install

Is necessary Python=3.8+

1. Clone repo

```shell
git clone https://github.com/johwconst/FaceQualityAssurance
```
2. Go to folder:
```shell
cd ./FaceQualityAssurance
```

3. Install requeriments:
```shell
pip install -r requeriments.txt
```

> [!TIP]
> Tip: Recommend installing dependencies in an isolated environment, such as using the [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.conda.io/en/latest/)!


## How Run
Create a python file in the project folder and add face_qa to the image you want to validate: 

```shell
from face_qa.face_qa import FaceQA

image_path = 'images/image_2.jpg'

face_qa = FaceQA(image_path, 1)

result = face_qa.check_face()
```

The result will be the following:

```shell
{
"face_detected": True, // Returns true if a face is detected
"more_than_one_face": False, // Returns true if more than one face is detected
"eyes_is_good": True, // Returns true if the quality of the eyes is good
"is_smiling": True, // Returns true if a smile is detected
"contrast_is_good": True, // Returns true if the contrast is good
"brightness_is_good": True, // Returns true if the brightness is good
"face_is_centralized": True // Returns true if the face is centralized
}
```

# Support or study materials
- <a href="https://pages.nist.gov/ifpc/2022/presentations/2_IFPC2022_OFIQ_Overview_Stratmann.pdf">Open Source Face Image Quality (OFIQ)
An Overview</a>
- <a href="https://pages.nist.gov/ifpc/2022/presentations/3_2022-11-07_OFIQ_SOTA.pdf">Facial Image Quality Assessment – State of the Art </a>

# Contributing!
Contributions are welcome! If you want to add a tool, fix a bug or improve the documentation, feel free to open an issue and/or make a pull request.
