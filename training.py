import logging
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, losses, metrics
from cnnmodel import CNNModel
from utils import read_yaml, logger, arg_parser


def main():
    parsed_args = arg_parser()
    params = read_yaml(parsed_args.params)

    logging_format = '%(asctime)s - %(message)s'
    datefmt = '%Y_%b_%d %X%z'
    logger(logging_format=logging_format, datefmt=datefmt)

    def make_generator(directory, shuffle: bool = False):
        output_generator = ImageDataGenerator().flow_from_directory(
            directory=directory,
            target_size=params['image_size'],
            class_mode='binary',
            shuffle=shuffle,
            seed=0,
        )
        return output_generator

    train_generator = make_generator(params['train_folder'], shuffle=True)
    val_generator = make_generator(params['validation_folder'])
    test_generator = make_generator(params['test_folder'])

    model = CNNModel()

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.BinaryCrossentropy(),
        metrics=[metrics.BinaryAccuracy()]
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=params['epochs'],
        batch_size=params['batch_size']
    )
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(1, params['epochs']+1))
    for i, j, epoch in zip(train_loss, val_loss, epochs):
        logging.info(f'Epoch: {epoch}, train loss: {i:.4f}, val loss: {j:.4f}')

    # disable logging here cause I don't need to log "model saved" info
    logging.disable(logging.INFO)
    model.save(params['model_name'], save_format='tf')

    predicts = model.evaluate(test_generator, verbose=0, return_dict=True)

    print('Test accuracy:', np.round(predicts['binary_accuracy'], 3))


if __name__ == '__main__':
    main()
