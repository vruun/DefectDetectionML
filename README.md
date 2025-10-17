# Defect Detection via ML + gRPC

**Author:** Varun Vaijnath 

## Project Overview

This project implements an automated defect detection system using **EfficientNet-B0** as the backbone model, and offers inference through a **gRPC-based client–server architecture**.  
The server hosts the model and handles prediction requests; the client sends image data and receives defect detection outputs.

## Motivation

- In industrial settings, detecting defects (scratches, dents, anomalies) in objects is time-consuming when done manually.  
- Automating this with a lightweight model + client-server interface can enable scalable deployment.  

## Architecture

- The client can be a GUI or CLI that sends images or image payloads.  
- The server handles loading the model, preprocessing, inference, postprocessing, and sends back the result.

## Technologies & Dependencies

- **Python** (version ≥ 3.x)  
- **gRPC / grpcio**  
- **TensorFlow** / **PyTorch** (or whichever framework you used)  
- **EfficientNet-B0** (pretrained / fine-tuned)  
- `numpy`, `opencv-python`, `protobuf`, etc.  

## Setup & Installation

1. Clone the repository:  
   
   git clone https://github.com/vruun/DefectDetectionML.git
   cd DefectDetectionML
   
3. Install Dependencies:
   
   pip install -r requirements.txt

## How to Run

1. Run the Server:
   
   cd ml_grpc_project
   python server.py
  
2. Run the client:
   
   cd ml_grpc_project
   python client.py testimgpath.jpg


