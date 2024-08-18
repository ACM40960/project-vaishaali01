# Exploring Advanced Transfer Learning Models for Effective Irish Sign Language Recognition

**Authors:**  
Vaishaali Kondapalli (23200337), Minu Jose (23200724)  
School of Mathematics and Statistics, University College Dublin

## Abstract

This project focuses on the detection of Irish Sign Language (ISL) gestures using advanced transfer learning-based object detection models. The study utilized a custom dataset, three transfer learning models (YOLOv8, YOLO-NAS, and RT-DETR), and a standard sequential CNN for comparative analysis. Key aspects of this evaluation include data augmentation, hyperparameter tuning, and performance metrics like <b>mAP50</b>. Among these, YOLO-NAS demonstrated superior accuracy and consistency, marking a significant step forward in ISL recognition technology.

**Note:** <b>It is highly recommended to run the code on Google Colab to avoid running into any type of dependency issues.</b>


## Open the FDS_Train_YOLONAS_Custom_Dataset_Sign_Language_Complete ipynb file in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LrL-Mh7U7eu4z4q6JKNw1FyW-pReNe_A?usp=sharing)

## Open RealTime Detection Transformer(RT-DETR) ipynb file in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Lpq5omFG8CKijPO9cJHYtq6qexHtkmwF?usp=sharing)
## Open YOLOv8 ipynb file in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c4fFqwiA4hFXdhwVJQyjsdMrl0Jsx8BT?usp=sharing)

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Data Collection and Preparation](#data-collection-and-preparation)
  - [Model Selection](#model-selection)
- [Results and Analysis](#results-and-analysis)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Introduction

Sign languages are crucial for communication within the deaf and hard-of-hearing communities, with Irish Sign Language (ISL) playing a key role in cultural identity and social inclusion. This project investigates the application of transfer learning-based models—YOLO-NAS, YOLOv8, and RT-DETR—for detecting ISL gestures.

By comparing these models to a standard sequential CNN and utilizing a custom dataset with image augmentation, the study aims to improve ISL recognition and contribute to its technological development.

## Methodology

### Data Collection and Preparation

- **Dataset Construction:** The dataset comprises seven phrases: *Hello, Help, Please, Thank You, Water, Sorry,* and *Phone Number*. It was constructed using OpenCV and manually annotated using the LabelImg program.
- **Image Count:** A total of 445 images were collected:
  - 282 images for training,
  - 138 images for validation,
  - 25 images for testing.
  
So overall images and annotations combined we have close to <b>1000</b> files.

- **Data Augmentation:** Various data augmentation techniques were applied during training, including random rotations, flips, scaling, and color changes.

**Dataset Access:** You can download the full dataset from the following link: [ISL Gesture Dataset](https://drive.google.com/drive/folders/1kgJX1HR1SkKsPgVGyOyxmCKWIvxuGyyA?usp=sharing).


The data is simply structured in a different way while running our sequential CNN model, You can download the full dataset for it from the following link: 
[ISL Gesture Dataset](https://drive.google.com/drive/folders/11OYto-wj4u9cKsZr5w2uzysLxdn9NwKN?usp=sharing).

### Model Selection

- **YOLO-NAS (S sized):** YOLO-NAS (S sized) model, developed using advanced Neural Architecture Search (NAS) technology, is chosen due to its high accuracy and remarkable inference speed. It outperforms previous YOLO versions like YOLOv8. Its quantization-aware architecture ensures efficient performance even in resource-constrained environments, making it ideal for specialized applications like ISL detection.
- **RT-DETR:** RT-DETR is a strong contender against YOLO models, particularly since YOLO models’ accuracy can be hindered by the use of Non-Maximum Suppression (NMS). RT-DETR outperforms the latest YOLOv8 model, prompting us to investigate whether DETRs can surpass the advanced YOLO detectors in both speed and accuracy by eliminating the delay introduced by NMS.
- **YOLOv8:** YOLOv8, with its advanced neural network architecture, is particularly well-suited for cases like ours, where both accuracy and speed are critical for effective sign language detection. Its design incorporates the latest innovations in object detection, making it a strong candidate for real-time applications, which is why we chose to experiment with it alongside other state-of-the-art models.

- **CNN:** Also experimented with a standard sequential CNN to evaluate its performance comparatively.

## Results and Analysis

### YOLO-NAS S Results
- The YOLO-NAS model demonstrated exceptional performance, achieving high confidence scores across various gestures. Its high recall rate of 1.0 and mAP50 of 0.997 underscore its effective gesture detection capabilities.
- Notably, the minimal gap between training and validation loss portrays an ideal case scenario result.
<img src="https://github.com/user-attachments/assets/003a2a9b-972b-4438-9c81-277990257054" width="480" height="480">
<img src="https://github.com/user-attachments/assets/c70f3d62-11ba-4867-927c-eb7a473628b9" width="480" height="480">
<img src="https://github.com/user-attachments/assets/e02c98a2-01c1-4435-aa2b-7e24f30b88aa" width="480" height="480">


### RT-DETR Results
- The mAP50 value for the RT-DETR model is 0.94. The confusion matrix points out that for “Water” and “ThankYou” classes the false positives are quite high compared to other classes.
- The training and validation loss portrayed a gradual dip in values.
<img src="https://github.com/user-attachments/assets/7eaa7887-4c04-439b-a0f9-2e9e13d795da" width="480" height="480">
<img src="https://github.com/user-attachments/assets/0159d174-e85a-4db1-932e-546366259437" width="480" height="480">
<img src="https://github.com/user-attachments/assets/813aed5f-60e5-4889-8f69-abc7de7431ca" width="480" height="480">
<img src="https://github.com/user-attachments/assets/8d0328fd-e990-4e86-a6c1-43f27d21ab5e" width="480" height="480">
<img src="https://github.com/user-attachments/assets/d0144d01-cbd4-46f9-881d-43404ef13bfd" width="480" height="480">

### YOLOv8 Results
- YOLO-NAS outperformed the other models, showing a consistent increase in mAP50, in contrast to the more unstable performance of RT-DETR and YOLOv8.
<img src="https://github.com/user-attachments/assets/13d9ca4d-13f4-4c39-b80c-c75bb9598d4c" width="480" height="480">
<img src="https://github.com/user-attachments/assets/6d0efc7d-5fa7-44a3-9fc0-2e5af5aa2acd" width="480" height="480">

### CNN Results
- In contrast, the sequential CNN model showed a tendency to overfit, frequently clas- sifying test images with visible fingers as "Hello," which suggests a need for more experimentation with the architecture and fine-tuning to enhance the model’s ability to generalize across different gestures.


Overall, YOLO-NAS outperforms the other models, as evidenced by its highest con- fidence scores for test image predictions and a consistently increasing mAP50 graph, in contrast to the more unstable performance of RT-DETR and YOLOv8, which exhibited fluctuating mAP scores across epochs.

## Future Work

Future improvements to this project could include more gestures and more diverse examples, hence improving the model’s generalisation and accuracy. Increasing the dataset size and class diversity may help the system progress from simple sign lan- guage detection to a packed sign language interpreter. In addition to testing with new models and fine-tuning techniques to improve the system’s robustness and flex- ibility, deployment strategies should be taken into account to make the system us- able in real-world applications.

## References

1. Yian Zhao et al. "DETRs Beat YOLOs on Real-time Object Detection." arXiv preprint arXiv:2304.08069, 2023.
2. Anusha Puchakayala et al. "American Sign Language Recognition using Deep Learning." Proceedings of the 7th International Conference on Computing Methodologies and Communication (ICCMC-2023), IEEE Xplore, 2023, ISBN: 978-1-6654-6408-6. DOI: 10.1109/ICCMC56507.2023.10084015.
3. K. Abhinand et al. "Malayalam Sign Language Identification using Finetuned YOLOv8 and Computer Vision Techniques." arXiv, arXiv:2405.06702v1 [cs.CL], 08 May 2024. License: CC BY 4.0.
4. A. Mishra et al. ISL recognition of emergency words using mediapipe, CNN, and LSTM, in: 2023 International Conference on Power Energy, Environment & Intelligent Control (PEEIC), IEEE, 2023, pp. 322–325.

## Lincense

There is no lincense to contribute to this project, however suggestions can be sent to either vakon2001@gmail.com OR minujose20@gmail.com

