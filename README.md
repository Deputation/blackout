# blackout
A simple and efficient tool that uses deep learning to identify and blackout faces in images.

This project was made to experiment with DNNs.

## How to Run
Enter a VENV:
```
python -m venv venv
```

Install dependencies:
```
pip install -r requirements.txt
```

Run:
```
python .\blackout_faces_cli.py .\input.jpg .\output.jpg .\deploy.prototxt.txt .\res10_300x300_ssd_iter_140000.caffemodel
```

## Face Detection Resources

Model: https://github.com/keyurr2/face-detection/blob/master/res10_300x300_ssd_iter_140000.caffemodel

Prototxt: https://github.com/keyurr2/face-detection/blob/master/deploy.prototxt.txt