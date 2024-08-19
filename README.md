<p>
  <img width="300" height ="200" src="https://github.com/user-attachments/assets/57148dfd-2a29-4a19-af8c-ecfd838a2618">

</p>

# Exploring Advanced Transfer Learning Models for Effective Irish Sign Language Recognition

**Authors:**  
Vaishaali Kondapalli (23200337), Minu Jose (23200724)  
School of Mathematics and Statistics, University College Dublin

## Abstract

This project focuses on the detection of Irish Sign Language (ISL) gestures using advanced transfer learning-based object detection models. The study utilized a custom dataset, three transfer learning models (YOLOv8, YOLO-NAS, and RT-DETR), and a standard sequential CNN for comparative analysis. Key aspects of this evaluation include data augmentation, hyperparameter tuning, and performance metrics like <b>mAP50</b>. Among these, YOLO-NAS demonstrated superior accuracy and consistency, marking a significant step forward in ISL recognition technology.

**Note:** <b>It is highly recommended to run the code on Google Colab to avoid running into any type of dependency issues. Also, because you're just given viewer access, the best way to go about it would be to create a copy of the file on your drive and then run it.</b>


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
  - [Model Selection Rationale](#model-selection-rationale)
  - [Models Overview](#models-overview)
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

**Dataset Access:** You can either create a copy/ download the full dataset from the following link: [ISL Gesture Dataset](https://drive.google.com/drive/folders/1kgJX1HR1SkKsPgVGyOyxmCKWIvxuGyyA?usp=sharing).


The data is simply structured in a different way while running our sequential CNN model, You can download the full dataset for it from the following link: 
[ISL Gesture Dataset](https://drive.google.com/drive/folders/11OYto-wj4u9cKsZr5w2uzysLxdn9NwKN?usp=sharing).


### Model Selection Rationale
Choosing the right model for detecting Irish Sign Language gestures required a balance of accuracy, speed, and efficiency. Here’s why each model was selected:

- **YOLO-NAS (S sized):** YOLO-NAS (S sized) model, developed using advanced Neural Architecture Search (NAS) technology, is chosen due to its high accuracy and remarkable inference speed. It outperforms previous YOLO versions like YOLOv8. Its quantization-aware architecture ensures efficient performance even in resource-constrained environments, making it ideal for specialized applications like ISL detection. 
- **RT-DETR:** RT-DETR is a strong contender against YOLO models, particularly since YOLO models’ accuracy can be hindered by the use of Non-Maximum Suppression (NMS). RT-DETR outperforms the latest YOLOv8 model, prompting us to investigate whether DETRs can surpass the advanced YOLO detectors in both speed and accuracy by eliminating the delay introduced by NMS.
- **YOLOv8:** YOLOv8, with its advanced neural network architecture, is particularly well-suited for cases like ours, where both accuracy and speed are critical for effective sign language detection. Its design incorporates the latest innovations in object detection, making it a strong candidate for real-time applications, which is why we chose to experiment with it alongside other state-of-the-art models.
- **CNN:** Also experimented with a standard sequential CNN to evaluate its performance comparatively.

### Models Overview
This section provides an in-depth look at each model used in our analysis :)

#### YOLO-NAS: Advanced Architecture for Superior Object Detection

**YOLO-NAS** is a next-generation object detection model developed by Deci AI, built upon the foundational YOLO series with significant enhancements. 

<p align="center">
  <img src="https://blog.paperspace.com/content/images/2024/01/Screenshot-2024-01-23-at-4.07.39-PM.png">
  Evolution of YOLO: A Foundational Object Detection Model
</p>

At its core, YOLO-NAS leverages **Neural Architecture Search (NAS)** technology through Deci's proprietary **AutoNAC** engine. This technology optimizes the model's architecture to achieve the best balance between accuracy and latency, making YOLO-NAS a state-of-the-art (SOTA) performer in real-time object detection.

One of the key innovations in YOLO-NAS is the use of **quantization-aware blocks** within the architecture. These blocks allow the model to support INT8 quantization, which converts neural network parameters from floating-point values to integer values, leading to greater efficiency with minimal loss in accuracy. This results in an exceptionally robust model that maintains high precision even when optimized for lower computational requirements.

<p align="center">
  <img src="https://github.com/vaishaalik/ISL-Recognition/blob/main/YOLO-NAS-architecture.png" width="640" height="640">
  <p align="center">
    YOLO-NAS Architecture
  </p>
</p>

#### Unparalleled Performance with AutoNAC Technology

YOLO-NAS is designed to outperform its predecessors by addressing common limitations such as insufficient quantization support and suboptimal accuracy-latency trade-offs. The **AutoNAC** technology plays a crucial role in this, enabling the model to be pre-trained on large-scale datasets like COCO and Objects365, and further refined through knowledge distillation and Distribution Focal Loss (DFL). This approach not only enhances the model's performance but also ensures it remains highly effective in diverse production environments.

With a superior mean average precision (mAP) YOLO-NAS surpasses previous models in the YOLO series, offering a perfect blend of speed and accuracy. Whether you’re working on large-scale object detection tasks or need a model optimized for edge devices, YOLO-NAS delivers cutting-edge performance that sets a new benchmark in the field of computer vision.

#### Real Time Detection Transformers(RT-DETR)
RT-DETR consists of a backbone, an efficient hybrid encoder, and a Transformer decoder with auxiliary prediction
heads. The overview of RT-DETR is illustrated in below figure. Specifically, we feed the features from the last three stages
of the backbone {S3,S4,S5} into the encoder. The efficient hybrid encoder transforms multi-scale features into a
sequence of image features through intra-scale feature interaction and cross-scale feature fusion. Subsequently, the uncertainty-minimal query selection is employed to select a fixed number of encoder features to serve as initial object queries for the decoder. Finally, the
decoder with auxiliary prediction heads iteratively optimizes object queries to generate categories and boxes.
<p align="center">
  <img src="https://github.com/user-attachments/assets/71902c05-0ece-417a-99bc-698c047095c5">
  <p align="center">
    RT-DETR Architecture
  </p>
</p>

#### YOLOv8 
YOLOv8 model incorporates advanced components like decoupled head architecture for improved detection performance and better feature representation.
<p align="center">
  <img src="https://github.com/user-attachments/assets/51b652f8-f52e-473a-aa84-3228f2effd06" width="640" height="640">
  <p align="center">
    YOLOv8 Architecture
  </p>
</p>

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
- In contrast, the sequential CNN model showed a tendency to overfit, frequently classifying test images with visible fingers as "Hello," which suggests a need for more experimentation with the architecture and fine-tuning to enhance the model’s ability to generalize across different gestures.


Overall, YOLO-NAS outperforms the other models, as evidenced by its highest confidence scores for test image predictions and a consistently increasing mAP50 graph, in contrast to the more unstable performance of RT-DETR and YOLOv8, which exhibited fluctuating mAP scores across epochs.

## Future Work

Future improvements to this project could include more gestures and more diverse examples, hence improving the model’s generalisation and accuracy. Increasing the dataset size and class diversity may help the system progress from simple sign lan- guage detection to a packed sign language interpreter. In addition to testing with new models and fine-tuning techniques to improve the system’s robustness and flexibility, deployment strategies should be taken into account to make the system us- able in real-world applications.

## References

1. Yian Zhao et al. "DETRs Beat YOLOs on Real-time Object Detection." arXiv preprint arXiv:2304.08069, 2023.
2. Anusha Puchakayala et al. "American Sign Language Recognition using Deep Learning." Proceedings of the 7th International Conference on Computing Methodologies and Communication (ICCMC-2023), IEEE Xplore, 2023, ISBN: 978-1-6654-6408-6. DOI: 10.1109/ICCMC56507.2023.10084015.
3. K. Abhinand et al. "Malayalam Sign Language Identification using Finetuned YOLOv8 and Computer Vision Techniques." arXiv, arXiv:2405.06702v1 [cs.CL], 08 May 2024. License: CC BY 4.0.
4. A. Mishra et al. ISL recognition of emergency words using mediapipe, CNN, and LSTM, in: 2023 International Conference on Power Energy, Environment & Intelligent Control (PEEIC), IEEE, 2023, pp. 322–325.
5. Xiangxiang Chu, Liang Li, Bo Zhang. "Make RepVGG Greater Again: A Quantization-aware Approach." arXiv preprint arXiv:2212.01593, submitted on 3 Dec 2022, last revised 11 Dec 2023. Available at: https://arxiv.org/abs/2212.01593
6. https://blog.paperspace.com/yolo-nas/

## License

There is no lincense to contribute to this project, however suggestions can be sent to either vakon2001@gmail.com OR minujose20@gmail.com

