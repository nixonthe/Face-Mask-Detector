import yaml
import argparse
import logging


def read_yaml(yaml_path):
    with open(yaml_path) as f:
        file = yaml.safe_load(f)
    return file


def logger(logging_format, datefmt, filemode='w'):
    logging.basicConfig(
        filename='train_log.log',
        level=logging.INFO,
        format=logging_format,
        datefmt=datefmt,
        filemode=filemode
    )


def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--params', '-p', default='params.yaml')
    parsed_args = args.parse_args()
    return parsed_args
