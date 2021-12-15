from mtcnn import MTCNN
import os
from tensorflow.keras.models import load_model
from camera import PhotoCamera
from utils import read_yaml, arg_parser


def main():
    parsed_args = arg_parser()
    params = read_yaml(parsed_args.params)

    images = [image for image in os.listdir('demo')]

    face_model = MTCNN()
    model = load_model(params['model_name'])

    camera = PhotoCamera(face_model, model)
    for image in images:
        path2img = os.path.join('demo', image)
        camera.predict_photo(path2img)


if __name__ == '__main__':
    main()
