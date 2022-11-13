# Project for the CV university course

The aim of this project is to develop an algorithm, written in C++, capable of:

- detecting human hands in an input image
- segmenting human hands in the image from the background

We are allowed to use CNNs only to solve one of the two tasks (for instance we cannot use  Mask R-CNN). For this reason the solution is based on:

- YOLOv4 for object detection
- a specific algorithm described in the pdf (_report.pdf_)

To compile the program, follow the instructions reported in _report.pdf_ (first paragraph: Introduction). Before, you have to download the weights for YOLOv4 by using this link

https://drive.google.com/file/d/10529N-OjnqlqDrhxdeygW73PKwJ1DX2H/view?usp=sharing

Then save this file (_yolo-obj_3000.weights_) in the directory _yolo_files_. In _report.pdf_ you find

- the program's tree, namely a description of the content of every folder
- how YOLOv4 has been trained (you may find the link of the dataset used)
- some notes about the code, in particular regarding the classes implemented
- the results obtained with some test images
