import os

import numpy as np
import pymia.data.assembler as pymia_asmbl
import pymia.data.conversion as conv
import pymia.deeplearning.training as train
import pymia.deeplearning.logging as log
import pymia.deeplearning.model as mdl
import SimpleITK as sitk
import tensorflow as tf

import pc.configuration.config as cfg
import pc.data.data as data
import pc.data.handler as hdlr
import pc.utilities.assembler as asmbl
import pc.utilities.evaluation as eval_
import pc.utilities.filesystem as fs
import pc.utilities.plotting_qualitative as qualitative


def validate_on_subject(self: train.Trainer, subject_assembler: pymia_asmbl.SubjectAssembler,
                        config: cfg.Configuration, is_training: bool) -> float:
    # prepare filesystem and evaluator
    if self.current_epoch % self.save_validation_nth_epoch == 0:
        epoch_result_dir = fs.prepare_epoch_result_directory(config.result_dir, self.current_epoch)
        epoch_csv_file = os.path.join(
            epoch_result_dir,
            '{}_{}{}.csv'.format(os.path.basename(config.result_dir), self.current_epoch,
                                 '_train' if is_training else ''))
        epoch_txt_file = os.path.join(
            epoch_result_dir,
            '{}_{}{}.txt'.format(os.path.basename(config.result_dir), self.current_epoch,
                                 '_train' if is_training else ''))
        evaluator = eval_.init_evaluator(not is_training, epoch_csv_file, True)
    else:
        epoch_csv_file = None
        epoch_result_dir = None
        epoch_txt_file = None
        evaluator = eval_.init_evaluator(not is_training)

    if not is_training:
        print('Epoch {:d}, {} s:'.format(self.current_epoch, self.epoch_duration))

    # loop over all subjects
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']
        label = subject_data['labels']
        label = np.squeeze(label, -1)

        probabilities = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.argmax(probabilities, -1)  # convert to class labels

        # Save predictions as SimpleITK images and save other images
        if self.current_epoch % self.save_validation_nth_epoch == 0:
            qualitative.PointCloudPlotter(epoch_result_dir).plot(subject_name, prediction, label,
                                                                 subject_data['indices'],
                                                                 subject_data['properties'])

            label_image = conv.NumpySimpleITKImageBridge.convert(subject_data['gt'], subject_data['properties'])
            prediction_image = data.point_cloud_to_image(prediction, subject_data['indices'], subject_data['properties'])
            evaluator.evaluate(prediction_image, label_image,
                               subject_name)  # use SimpleITK images for Hausdorff distance

            if not is_training:
                subject_results = os.path.join(epoch_result_dir, subject_name)
                os.makedirs(subject_results, exist_ok=True)

                # save predicted point cloud
                data.save_to_csv(os.path.join(subject_results, '{}_PREDICTION.csv'.format(subject_name)),
                                 subject_data['indices'][prediction == 1], None, subject_data['properties'])

                data.save_to_csv(os.path.join(subject_results, '{}_PROBABILITY.csv'.format(subject_name)),
                                 subject_data['indices'], probabilities[:, 1:2], subject_data['properties'])

                # save predicted segmentation
                sitk.WriteImage(prediction_image,
                                os.path.join(subject_results, '{}_PREDICTION.mha'.format(subject_name)),
                                True)

                probabilities_image = data.point_cloud_to_image(probabilities, subject_data['indices'], subject_data['properties'])
                sitk.WriteImage(probabilities_image,
                               os.path.join(subject_results, '{}_PROBABILITY.mha'.format(subject_name)), True)
        else:
            evaluator.evaluate(prediction, label, subject_name)

    # log to TensorBoard
    results = eval_.aggregate_results(evaluator)
    score = 0
    for result in results:
        self.logger.log_scalar('{}/MEAN'.format(result.metric), result.mean, self.current_epoch,
                               is_training)
        self.logger.log_scalar('{}/STD'.format(result.metric), result.std, self.current_epoch,
                               is_training)
        if result.metric == 'DICE':
            score = result.mean  # score for best model

    print('Aggregated {} results (epoch {:d}):'.format('training' if is_training else 'validation', self.current_epoch))
    if self.current_epoch % self.save_validation_nth_epoch == 0:
        eval_.AggregatedResultWriter(epoch_txt_file).write(results)
        # stat.QuantitativePlotter(epoch_result_dir).plot(epoch_csv_file, 'train' if is_training else None)
    else:
        eval_.AggregatedResultWriter().write(results)

    return score if not is_training else -1


class SegmentationTrainer(train.TensorFlowTrainer):

    def __init__(self, data_handler: hdlr.PointCloudDataHandler, logger: log.TensorFlowLogger,
                 config: cfg.Configuration, model: mdl.TensorFlowModel, session: tf.Session):
        super().__init__(data_handler, logger, config, model, session)
        self.config = config

    def init_subject_assembler(self) -> pymia_asmbl.Assembler:
        return asmbl.init_subject_assembler()

    def validate_on_subject(self, subject_assembler: pymia_asmbl.SubjectAssembler, is_training: bool) -> float:
        if is_training:
            # we could directly use the subject_assembler variable but the assembled subjects will be of augmented data,
            # and therefore calculating a e.g. Dice coefficient will give non-meaningful results, therefore,
            # predict the subjects in the training set subjects newly
            subject_assembler = self.init_subject_assembler()
            self.data_handler.dataset.set_extractor(self.data_handler.extractor_valid)
            self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_valid)
            for batch_idx, batch in enumerate(self.data_handler.loader_train):
                prediction, _ = self.validate_batch(batch_idx, batch)
                subject_assembler.add_batch(prediction, batch)

            validate_on_subject(self, subject_assembler, self.config, is_training)
            return -1
        else:
            return validate_on_subject(self, subject_assembler, self.config, is_training)

    def set_seed(self):
        super().set_seed()
        self.data_handler.point_cloud_shuffler.shuffle()

    def batch_to_feed_dict(self, batch: dict, is_training: bool):
        """Generates the TensorFlow feed dictionary.

        Args:
            batch: The batch from the data loader.
            is_training: Indicates whether it is the training or testing phase.

        Returns:
            The feed dictionary.
        """
        feed_dict = {self.model.x_placeholder: np.stack(batch['images'], axis=0),
                     self.model.y_placeholder: np.stack(batch['labels'], axis=0),
                     self.model.is_training_placeholder: is_training}

        if self.config.use_image_information:
            feed_dict[self.model.image_information_placeholder] = np.stack(batch[data.KEY_IMAGE_INFORMATION], axis=0)

        return feed_dict

    def train_batch(self, idx, batch: dict):
        prediction, loss_val = super().train_batch(idx, batch)
        if isinstance(prediction, tuple):
            return prediction[0], loss_val
        else:
            return prediction, loss_val

    def validate_batch(self, idx: int, batch: dict):
        prediction, loss_val = super().validate_batch(idx, batch)
        if isinstance(prediction, tuple):
            return prediction[0], loss_val
        else:
            return prediction, loss_val


class AssemblingTester(SegmentationTrainer):
    """Use this class to test the training/validation without a network. The metrics should have the maximum values.

    Beware of possible data augmentation during testing!
    """

    def _get_labels_in_tensorflow_format(self, batch):
        feed_dict = self.batch_to_feed_dict(batch, True)
        labels = np.expand_dims(feed_dict[self.model.y_placeholder], -1)
        labels = np.concatenate([1 - labels, labels], -1)
        return labels

    def train_batch(self, idx, batch: dict):
        return self._get_labels_in_tensorflow_format(batch), 0.0

    def validate_batch(self, idx: int, batch: dict) -> (np.ndarray, float):
        return self._get_labels_in_tensorflow_format(batch), 0.0
