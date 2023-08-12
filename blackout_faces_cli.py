import cv2
import numpy as np
import argparse
import os
import requests

def blackout(image_path, output_path, prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 0), -1)
    
    cv2.imwrite(output_path, image)

def main():
    parser = argparse.ArgumentParser(description='Blackout faces in an image using DNN-based face detection.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    parser.add_argument('output_image', type=str, help='Path to save the output image with faces blacked out.')
    parser.add_argument('prototxt_path', type=str, help='Path to the DNN prototxt configuration file.')
    parser.add_argument('model_path', type=str, help='Path to the DNN model file.')
    args = parser.parse_args()

    blackout(args.input_image, args.output_image, args.prototxt_path, args.model_path)

if __name__ == '__main__':
    main()