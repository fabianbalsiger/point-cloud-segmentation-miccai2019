import numpy as np

import pymia.data.definition as df
import pymia.data.extraction as ext
import pymia.data.transformation as tfm

import pc.data.data as d


class PointCloudShuffler:

    def __init__(self, sizes):
        self.sizes = sizes
        self.selection_arrays = {}
        self.shuffle()

    def shuffle(self):
        self.selection_arrays = {}
        for subject, n_points in self.sizes.items():
            indices = np.arange(n_points)
            np.random.shuffle(indices)
            self.selection_arrays[subject] = indices


class ShuffledDataExtractor(ext.Extractor):

    def __init__(self, shuffler: PointCloudShuffler, categories=('images',)) -> None:
        super().__init__()
        self.shuffler = shuffler
        self.categories = categories
        self.entry_base_names = None
        self.image_information = {}
        self.image_information_labels = {}

    def extract(self, reader: ext.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']
        index_expr = params['index_expr']

        base_name = self.entry_base_names[subject_index]
        for category in self.categories:
            if category == d.KEY_IMAGE_INFORMATION:
                data = self.get_image_information(reader, base_name)
            elif category == d.KEY_IMAGE_INFORMATION_LABELS:
                data = self.get_image_information_labels(reader, base_name)
            else:
                data = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(category), base_name))
            data = data[self.shuffler.selection_arrays[subject_index]]
            extracted[category] = data[index_expr.expression]

    def get_image_information(self, reader, base_name):
        if base_name not in self.image_information:
            self.image_information[base_name] = reader.read('{}/{}'.format(df.DATA_PLACEHOLDER.format(d.KEY_IMAGE_INFORMATION), base_name))
        return self.image_information[base_name]

    def get_image_information_labels(self, reader, base_name):
        if base_name not in self.image_information_labels:
            self.image_information_labels[base_name] = reader.read('{}/{}'.format(
                df.DATA_PLACEHOLDER.format(d.KEY_IMAGE_INFORMATION_LABELS), base_name))  # we squeeze due to dataset convention
        return self.image_information_labels[base_name]


class PointCloudRotate(tfm.Transform):

    def __init__(self, entries=('images', )):
        """Randomly rotates a point cloud.

        Args:
            p (float): The probability of the mirroring to be applied.
            entries (tuple): The sample's entries to apply the mirroring to.
        """
        super().__init__()
        # self.p = p
        self.entries = entries

    def __call__(self, sample: dict) -> dict:
        for entry in self.entries:
            if entry not in sample:
                raise ValueError(tfm.ENTRY_NOT_EXTRACTED_ERR_MSG.format(entry))

            angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(angle)
            sinval = np.sin(angle)
            rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])  # R_z

            sample[entry] = np.dot(sample[entry], rotation_matrix).astype(np.float32)  # todo handle possible additional features

        return sample


class PointCloudJitter(tfm.Transform):

    def __init__(self, std=0.01, clip=0.03, entries=('images', )):
        self.std = std
        self.clip = clip
        self.entires = entries

    def __call__(self, sample: dict):

        jitter = np.random.normal(0.0, self.std, sample['images'].shape)
        jitter = np.clip(jitter, -self.clip, self.clip)

        sample['images'][:, 0:2] += jitter[:, 0:2]

        return sample
