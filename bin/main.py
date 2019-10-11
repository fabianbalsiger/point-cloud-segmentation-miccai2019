import argparse
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import pymia.deeplearning.logging as log
import tensorflow as tf

import pc.configuration.config as cfg
import pc.data.handler as hdlr
import pc.data.split as split
import pc.model.point_cnn as net
import pc.utilities.filesystem as fs
import pc.utilities.seeding as seed
import pc.utilities.training as train


def main(config_file: str):
    config = cfg.load(config_file, cfg.Configuration)

    # set up directories and logging
    model_dir, result_dir = fs.prepare_directories(config_file, cfg.Configuration,
                                                   lambda: fs.get_directory_name(config))
    config.model_dir = model_dir
    config.result_dir = result_dir
    print(config)

    # set seed before model instantiation
    print('Set seed to {}'.format(config.seed))
    seed.set_seed(config.seed, config.cudnn_determinism)

    # load train and valid subjects from split file
    subjects_train, subjects_valid, _ = split.load_split(config.split_file)
    print('Train subjects:', subjects_train)
    print('Valid subjects:', subjects_valid)

    # set up data handling
    data_handler = hdlr.PointCloudDataHandler(config, subjects_train, subjects_valid, None)

    with tf.Session() as sess:
        # extract a sample for model initialization
        data_handler.dataset.set_extractor(data_handler.extractor_train)
        data_handler.dataset.set_transform(data_handler.extraction_transform_train)
        sample = data_handler.dataset[0]

        model = net.PointCNN(sess, sample, config)
        logger = log.TensorFlowLogger(config.model_dir, sess,
                                      model.epoch_summaries(),
                                      model.batch_summaries(),
                                      model.visualization_summaries())

        # trainer = train.AssemblingTester(data_handler, logger, config, model, sess)
        trainer = train.SegmentationTrainer(data_handler, logger, config, model, sess)

        tf.get_default_graph().finalize()  # to ensure that no ops are added during training, which would lead to
        # a growing graph
        trainer.train()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Deep learning for shape learning on point clouds')

    parser.add_argument(
        '--config_file',
        type=str,
        default='./bin/config.json',
        help='Path to the configuration file.'
    )

    args = parser.parse_args()
    main(args.config_file)
