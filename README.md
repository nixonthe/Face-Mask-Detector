
# Face Mask Detector

This repo contain a full deep learning project with instruction how to run it on your PC.
You can train simple CNN neural network and detect some faces by using your own webcam!




## Demo

![Figure_1](https://user-images.githubusercontent.com/64987384/146253420-7478d937-abdb-412e-ba58-24e2e6dc7055.png)


## Link to original dataset

[Face Mask Detection ~12K Images Dataset](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset)


# Steps to run project

## Step 1:

Clone the project

```bash
  git clone https://github.com/nixonthe/Face-Mask-Detector.git
```

## Step 2:

Install requirements

```bash
  pip install -r requirements.txt
```

## Step 3:

Move downloaded dataset to the project folder. Rename dataset folder to **'data'** since its mentioned in params file

## Step 4:

To work with images you need to install SciPy module

```bash
  pip install scipy
```

## Step 5:

Run training module

```bash
  python training.py
```

Also you can easily run training with your own params. Just follow my template

```bash
  python training.py -p my_params.yaml
```
## Step 6:

To launch realtime face detection run next command

```bash
  python run_video.py
```

Enjoy!


## Feedback

If you have any feedback, please reach out to me at glazunovnik@gmail.com

