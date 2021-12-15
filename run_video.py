from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from camera import VideoCamera
from utils import read_yaml, arg_parser


def main():
    parsed_args = arg_parser()
    params = read_yaml(parsed_args.params)

    face_model = MTCNN()
    model = load_model(params['model_name'])

    camera = VideoCamera(face_model, model)
    camera.predict_stream()


if __name__ == '__main__':
    main()
