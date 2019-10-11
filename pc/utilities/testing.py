import os

import numpy as np
import pymia.data.assembler as pymia_asmbl
import pymia.data.conversion as pymia_conv
import pymia.deeplearning.model as mdl
import pymia.deeplearning.testing as test
import SimpleITK as sitk
import tensorflow as tf

import pc.configuration.config as cfg
import pc.data.data as data
import pc.data.handler as hdlr
import pc.utilities.assembler as asmbl
import pc.utilities.evaluation as eval_
import pc.utilities.plotting_qualitative as qualitative


def process_predictions(self: test.Tester, subject_assembler: pymia_asmbl.Assembler, result_dir, prediction_idx: int,
                        split_no : int = 0):
    os.makedirs(result_dir, exist_ok=True)
    csv_file = os.path.join(result_dir, 'results_S{}_{}.csv'.format(split_no, prediction_idx))

    if prediction_idx == 0:
        prediction_idx = ''

    evaluator = eval_.init_evaluator(True, csv_file, True)

    # loop over all subjects
    for subject_idx in list(subject_assembler.predictions.keys()):
        subject_data = self.data_handler.dataset.direct_extract(self.data_handler.extractor_test, subject_idx)
        subject_name = subject_data['subject']
        label = subject_data['labels']
        label = np.squeeze(label, -1)

        probabilities = subject_assembler.get_assembled_subject(subject_idx)
        prediction = np.argmax(probabilities, -1)  # convert to class labels

        qualitative.PointCloudPlotter(result_dir).plot(subject_name, prediction, label,
                                                       subject_data['indices'],
                                                       subject_data['properties'])

        label_image = pymia_conv.NumpySimpleITKImageBridge.convert(subject_data['gt'], subject_data['properties'])
        prediction_image = data.point_cloud_to_image(prediction, subject_data['indices'], subject_data['properties'])
        evaluator.evaluate(prediction_image, label_image,
                           subject_name)  # use SimpleITK images for Hausdorff distance

        subject_results = os.path.join(result_dir, subject_name)
        os.makedirs(subject_results, exist_ok=True)

        # save predicted segmentation
        sitk.WriteImage(prediction_image,
                        os.path.join(subject_results, '{}_PREDICTION{}.mha'.format(subject_name, prediction_idx)),
                        True)

        probabilities_image = data.point_cloud_to_image(probabilities, subject_data['indices'],
                                                        subject_data['properties'])
        sitk.WriteImage(probabilities_image,
                        os.path.join(subject_results, '{}_PROBABILITY{}.mha'.format(subject_name, prediction_idx)), True)


class TensorFlowTester(test.TensorFlowTester):

    def __init__(self, data_handler: hdlr.PointCloudDataHandler, model: mdl.TensorFlowModel, model_dir: str, result_dir: str,
                 config: cfg.Configuration, split_no: int, session: tf.Session, no_predictions: int = 5):
        super().__init__(data_handler, model, model_dir, session)
        self.result_dir = result_dir
        self.config = config
        self.split_no = split_no
        self.no_predictions = no_predictions
        self.current_prediction = 0

    def init_subject_assembler(self) -> pymia_asmbl.Assembler:
        return asmbl.init_subject_assembler()

    def process_predictions(self, subject_assembler: pymia_asmbl.Assembler):
        process_predictions(self, subject_assembler, self.result_dir, self.current_prediction, self.split_no)

    def batch_to_feed_dict(self, batch: dict):
        feed_dict = {self.model.x_placeholder: np.stack(batch['images'], axis=0),
                     self.model.y_placeholder: np.stack(batch['labels'], axis=0),
                     self.model.is_training_placeholder: False}

        if self.config.use_image_information:
            feed_dict[self.model.image_information_placeholder] = np.stack(batch[data.KEY_IMAGE_INFORMATION], axis=0)

        return feed_dict

    def predict(self):
        """Predicts subjects based on a pre-trained model.

        This is the main method, which will call the other methods.
        """

        if not self.is_model_loaded:
            self.load()

        subject_assembler = self.init_subject_assembler()  # todo: not optimal solution since it keeps everything in memory
        self.data_handler.dataset.set_extractor(self.data_handler.extractor_valid)
        self.data_handler.dataset.set_transform(self.data_handler.extraction_transform_valid)
        for prediction_idx in range(self.no_predictions):
            self.current_prediction = prediction_idx
            for batch_idx, batch in enumerate(self.data_handler.loader_test):
                prediction = self.predict_batch(batch_idx, batch)
                subject_assembler.add_batch(prediction, batch)
            self.process_predictions(subject_assembler)
            self.data_handler.point_cloud_shuffler_valid.shuffle()
