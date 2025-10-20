# Real-Time Object Detection & Tracking

This project implements a **real-time object detection and tracking system** using **YOLOv8**.  
The model is fine-tuned on a custom dataset with four classes — *person, bottle, cell phone,* and *spoon* — achieving **over 75% detection accuracy** on the validation set.

---

##  Project Overview
The goal of this project is to detect and track multiple objects in real time using video streams or webcam feeds.  
It leverages **Ultralytics YOLOv8** for detection and **OpenCV** for visualization and tracking.

---

##  Key Features
-  Fine-tuned **YOLOv8 model** for high-accuracy detection  
-  **Real-time object detection & tracking** using OpenCV  
-  Supports both **image and video inference**  
-  Achieves **>75% mean Average Precision (mAP)**  
-  Modular code design for easy customization and experimentation  

---

## Directory Structure
Real_Time_Object_Detection_Tracking/
│
├── data/ # Dataset (training/validation images)
├── models/ # YOLOv8 weights (best.pt)
├── src/ # Source code modules (detector, stream, tracking)
├── runs/ # YOLOv8 training runs and results
├── notebooks/ # Jupyter notebooks for training/inference
├── outputs/ # Inference results and visualizations
└── README.md # Project overview and documentation

##  Model Details
- **Framework:** Ultralytics YOLOv8  
- **Training Epochs:** 50  
- **Accuracy:** >75% mAP on validation set  
- **Classes:** Person, Bottle, Cell Phone, Spoon  
