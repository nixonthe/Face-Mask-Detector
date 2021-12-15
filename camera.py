import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import read_yaml, arg_parser

parsed_args = arg_parser()
params = read_yaml(parsed_args.params)


class VideoCamera:

    def __init__(self, face_detector, cnn_model, src=0):
        self.video = cv2.VideoCapture(src)
        self.face_detector = face_detector
        self.cnn_model = cnn_model
        self.labels = {0: 'MASK', 1: 'NO MASK'}

    def predict_stream(self):
        while True:
            ret, frame = self.video.read()
            faces = self.face_detector.detect_faces(frame)
            new_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]['box']
                cropped_image = new_image[y:y + h, x:x + w]
                resized_image = cv2.resize(cropped_image, params['image_size'])
                reshaped_image = np.reshape(resized_image,
                                            [1, params['image_size'][0], params['image_size'][1], 3]) / 255.0
                predicted = self.cnn_model.predict(reshaped_image)
                output = np.where(predicted > .5, 1, 0)[0][0]
                cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(new_image, self.labels[output], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
                cv2.imshow('Video', new_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()


class PhotoCamera:

    def __init__(self, face_detector, cnn_model):
        self.face_detector = face_detector
        self.cnn_model = cnn_model
        self.labels = {0: 'MASK', 1: 'NO MASK'}

    def predict_photo(self, image):
        pic = cv2.imread(image)
        faces = self.face_detector.detect_faces(pic)
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]['box']
            cropped_image = pic[y:y + h, x:x + w]
            resized_image = cv2.resize(cropped_image, (224, 224))
            reshaped_image = np.reshape(resized_image,
                                        [1, 224, 224, 3]) / 255.0
            predicted = self.cnn_model.predict(reshaped_image)
            output = np.where(predicted > .5, 1, 0)[0][0]
            cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(pic, self.labels[output], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 2)
        rgb_image = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_image)
        plt.show()
