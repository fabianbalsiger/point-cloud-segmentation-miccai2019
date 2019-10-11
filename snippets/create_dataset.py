import argparse
import glob
import os
import typing

import numpy as np
import pymia.data as pymia_data
import pymia.data.conversion as conv
import pymia.data.creation as pymia_crt
import pymia.data.transformation as pymia_tfm
import SimpleITK as sitk

import pc.data.data as data


class LoadData(pymia_crt.Load):

    def __init__(self, probability_threshold: float, spatial_size: int):
        self.probability_threshold = probability_threshold
        self.spatial_size = spatial_size
        self.probabilities = None
        self.gt_arr = None
        self.labels = None
        self.indices = None
        self.properties = None

    def __call__(self, file_name: str, id_: str, category: str, subject_id: str) -> \
            typing.Tuple[np.ndarray, typing.Union[conv.ImageProperties, None]]:
        # note that this function works only if the COORDINATE-case is executed before all others

        if id_ == data.FileTypes.COORDINATE.name:
            # note that the "---" is an ugly hack to be able to pass two paths...
            probability_file, gt_file = file_name.split('---')

            # get properties
            img_gt = sitk.ReadImage(gt_file)
            self.properties = conv.ImageProperties(img_gt)

            self.gt_arr = sitk.GetArrayFromImage(img_gt)

            # get probabilities
            img_proba = sitk.ReadImage(probability_file)
            proba_arr = sitk.GetArrayFromImage(img_proba)
            proba_arr = proba_arr[..., 1]  # get probabilities of foreground class

            # get indices of probabilites larger some value
            # meaning construct the point cloud
            proba_indices = np.nonzero(proba_arr > self.probability_threshold)
            proba_indices = np.array(proba_indices[::-1])
            proba_indices = np.transpose(proba_indices)  # shape is (N, 3)
            self.indices = proba_indices

            self.probabilities = np.zeros((self.indices.shape[0], 1), np.float32)  # shape is (N, 1)
            self.labels = np.zeros((self.indices.shape[0], 1), np.uint8)  # shape is (N, 1)

            for idx, (x, y, z) in enumerate(self.indices):
                self.probabilities[idx] = proba_arr[z, y, x]
                self.labels[idx] = self.gt_arr[z, y, x]

            return self.indices, self.properties
        elif id_ == data.FileTypes.INDICES.name:  # todo: needed
            return self.indices, self.properties
        elif id_ == data.FileTypes.LABEL.name:
            return self.labels, self.properties
        elif id_ == data.FileTypes.GTM.name:
            return self.gt_arr, self.properties
        elif id_ == data.FileTypes.IMAGE_INFORMATION.name:
            img = sitk.ReadImage(file_name)
            img = sitk.GetArrayFromImage(img)
            img = img[..., 1]  # get probabilities of foreground class
            img_arr = img.copy()
            img_arr = np.expand_dims(img_arr, -1)

            cubes = self._extract_cubes(img_arr)
            return cubes.astype(np.float32), self.properties
        else:
            raise ValueError('id "{}" unknown'.format(id_))

    def _extract_cubes(self, img_arr):
        # pad image to perform secure slicing
        pad_size = (self.spatial_size - 1) // 2
        img_arr = np.pad(img_arr, ((pad_size, pad_size), (pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                         'constant', constant_values=0)
        # allocate patch array
        cubes = np.zeros((self.indices.shape[0], self.spatial_size, self.spatial_size, self.spatial_size, img_arr.shape[-1]))
        for idx, (x, y, z) in enumerate(self.indices):
            cube = img_arr[z:(z + self.spatial_size), y:(y + self.spatial_size), x:(x + self.spatial_size)]
            cubes[idx] = cube
        return cubes


class Collector:

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.subject_files = []

        self._collect()

    def get_subject_files(self) -> typing.List[pymia_data.SubjectFile]:
        return self.subject_files

    def _collect(self):
        self.subject_files.clear()

        subject_dirs = glob.glob(os.path.join(self.root_dir, '*'))

        subject_dirs = list(filter(lambda path: os.path.basename(path).lower().startswith('subject')
                                                and os.path.isdir(path),
                                   subject_dirs))
        subject_dirs.sort(key=lambda path: os.path.basename(path))

        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)

            # we generate an entry for the coordinates of the points in our clouds
            # note that the "---" is an ugly hack to be able to pass two paths
            images = {data.FileTypes.COORDINATE.name:
                          os.path.join(subject_dir, '{}_PROBABILITY.mha'.format(subject)) +
                          '---' +
                          os.path.join(subject_dir, '{}_GROUND_TRUTH.mha'.format(subject))
                      }
            # we create an entry for the labels of each point
            labels = {data.FileTypes.LABEL.name:
                          os.path.join(subject_dir, '{}_GROUND_TRUTH.mha'.format(subject))}

            indices = {data.FileTypes.INDICES.name:
                           os.path.join(subject_dir, '{}_PROBABILITY.mha'.format(subject))}

            image_information = {data.FileTypes.IMAGE_INFORMATION.name:
                                     os.path.join(subject_dir, '{}_PROBABILITY.mha'.format(subject))}

            # we also save the ground truth in image format for easier evaluation
            gt = {data.FileTypes.GTM.name:
                      os.path.join(subject_dir, '{}_GROUND_TRUTH.mha'.format(subject))}

            sf = pymia_data.SubjectFile(subject, images=images, labels=labels,
                                        indices=indices,
                                        image_information=image_information,
                                        gt=gt
                                        )
            self.subject_files.append(sf)


def normalize_unit_cube(arr: np.ndarray):
    new_min_val = -1
    new_max_val = 1
    max_val = arr.max()
    min_val = arr.min()
    rescaled_np_img = (arr - min_val) / (max_val - min_val) * (new_max_val - new_min_val) + new_min_val
    return rescaled_np_img


def concat(data: typing.List[np.ndarray]) -> np.ndarray:
    if len(data) == 1:
        return data[0]
    return np.stack(data, axis=-1)


def create_sample_data(dir: str, no_subjects: int = 8):
    img_gt = sitk.ReadImage(os.path.join(dir, 'EXAMPLE', 'EXAMPLE_GROUND_TRUTH.mha'))
    img_proba = sitk.ReadImage(os.path.join(dir, 'EXAMPLE', 'EXAMPLE_PROBABILITY_CNN.mha'))
    size = img_proba.GetSize()

    for i in range(no_subjects):
        subject = 'Subject_{}'.format(i)
        subject_path = os.path.join(dir, subject)
        os.makedirs(subject_path, exist_ok=True)

        # copy the example ground truth
        sitk.WriteImage(img_gt, os.path.join(dir, subject, subject + '_GROUND_TRUTH.mha'), True)
        # copy the example probability output of the CNN
        sitk.WriteImage(img_proba, os.path.join(subject_path, subject + '_PROBABILITY.mha'), True)

        # create a random MR image (we cannot share the actual MR image)
        arr = np.random.rand(*size[::-1])
        img_mr = sitk.GetImageFromArray(arr)
        img_mr.CopyInformation(img_proba)
        sitk.WriteImage(img_mr, os.path.join(subject_path, subject + '_MR.mha'), True)


def main(hdf_file: str, data_dir: str):
    if os.path.exists(hdf_file):
        raise RuntimeError('Dataset file "{}" does exist already'.format(hdf_file))

    # we threshold I_Q at probability 0.1
    probability_threshold = 0.1
    # we use image information extracted from 5^3 neighborhood around each point
    spatial_size = 5

    # let's create some sample data
    np.random.seed(42)  # to have same sample data
    create_sample_data(data_dir, no_subjects=8)

    # collect the data
    collector = Collector(data_dir)
    subjects = collector.get_subject_files()

    for subject in subjects:
        print(subject.subject)
    print('Total of {} subjects'.format(len(subjects)))

    os.makedirs(os.path.dirname(hdf_file), exist_ok=True)

    with pymia_crt.get_writer(hdf_file) as writer:
        callbacks = pymia_crt.get_default_callbacks(writer)

        transform = pymia_tfm.ComposeTransform([
            pymia_tfm.LambdaTransform(lambda_fn=lambda np_data: np_data.astype(np.float32),
                                      entries=('images',
                                               data.KEY_IMAGE_INFORMATION,
                                               )),
            pymia_tfm.LambdaTransform(loop_axis=1, entries=('images', ), lambda_fn=normalize_unit_cube),
            pymia_tfm.IntensityNormalization(loop_axis=-1, entries=(data.KEY_IMAGE_INFORMATION,))
        ])

        traverser = pymia_crt.SubjectFileTraverser()
        traverser.traverse(subjects, callback=callbacks, load=LoadData(probability_threshold, spatial_size),
                           transform=transform, concat_fn=concat)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Dataset creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='../data',
        help='Path to the data directory.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.data_dir)
