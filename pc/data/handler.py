import typing

import numpy as np
import pymia.data.definition as pymia_df
import pymia.data.extraction as pymia_extr
import pymia.data.indexexpression as pymia_expr
import pymia.data.transformation as pymia_tfm
import pymia.deeplearning.conversion as pymia_cnv
import pymia.deeplearning.data_handler as hdlr

import pc.configuration.config as cfg
import pc.data.data as data
import pc.utilities.augmentation as aug


class PointCloudIndexing(pymia_extr.IndexingStrategy):

    def __init__(self, no_points: int = 1024):
        """Initializes a new instance of the PointCloudIndexing class.

        Args:
            no_points (int): The number of points to sample.
        """
        self.shape = None
        self.indexing = None
        self.no_points = no_points

    def __call__(self, shape) -> typing.List[pymia_expr.IndexExpression]:
        if self.shape == shape:
            return self.indexing

        self.shape = shape  # save for later comparison to avoid calculating indices if the shape is equal
        size = shape[0]
        if size < self.no_points:
            raise ValueError('Shape of size {} contains not {} point'.format(size, self.no_points))

        self.indexing = []

        for idx in range(0, self.no_points * (size // self.no_points), self.no_points):
            # do expression
            self.indexing.append(pymia_expr.IndexExpression((idx, idx + self.no_points)))
        self.indexing.append(pymia_expr.IndexExpression((size - self.no_points, size)))  # will overlap with last added

        return self.indexing


class PointCloudSizeExtractor(pymia_extr.Extractor):

    def __init__(self, categories=('images',)) -> None:
        super().__init__()
        self.categories = categories
        self.entry_base_names = None

    def extract(self, reader: pymia_extr.Reader, params: dict, extracted: dict) -> None:
        if self.entry_base_names is None:
            entries = reader.get_subject_entries()
            self.entry_base_names = [entry.rsplit('/', maxsplit=1)[1] for entry in entries]

        subject_index = params['subject_index']

        base_name = self.entry_base_names[subject_index]
        data = reader.read('{}/{}'.format(pymia_df.DATA_PLACEHOLDER.format(pymia_df.KEY_IMAGES), base_name))
        extracted['size'] = data.shape[0]


class ConcatenateCoordinatesAndPointFeatures(pymia_tfm.Transform):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: dict) -> dict:
        sample[pymia_df.KEY_IMAGES] = np.concatenate([sample[pymia_df.KEY_IMAGES], sample[data.KEY_POINT_FEATURES]], -1)
        return sample


class PointCloudDataHandler(hdlr.DataHandler):

    def __init__(self, config: cfg.Configuration,
                 subjects_train,
                 subjects_valid,
                 subjects_test,
                 collate_fn=pymia_cnv.TensorFlowCollate()):
        super().__init__()

        indexing_strategy = PointCloudIndexing(config.no_points)

        self.dataset = pymia_extr.ParameterizableDataset(config.database_file,
                                                         indexing_strategy,
                                                         pymia_extr.SubjectExtractor(),  # for the usual select_indices
                                                         None)

        self.no_subjects_train = len(subjects_train)
        self.no_subjects_valid = len(subjects_valid)
        self.no_subjects_test = len(subjects_valid)  # same as validation for this kind of cross validation

        # get sampler ids by subjects
        sampler_ids_train = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_train))
        sampler_ids_valid = pymia_extr.select_indices(self.dataset, pymia_extr.SubjectSelection(subjects_valid))

        categories = ('images', 'labels')
        categories_tfm = ('images', 'labels')
        if config.use_image_information:
            image_information_categories = (data.KEY_IMAGE_INFORMATION, )
            categories += image_information_categories
            categories_tfm += image_information_categories
            collate_fn.entries += image_information_categories

        # define point cloud shuffler for augmentation
        sizes = {}
        for idx in range(len(self.dataset.get_subjects())):
            sample = self.dataset.direct_extract(PointCloudSizeExtractor(), idx)
            sizes[idx] = sample['size']
        self.point_cloud_shuffler = aug.PointCloudShuffler(sizes)
        self.point_cloud_shuffler_valid = aug.PointCloudShuffler(sizes)  # will only shuffle once at instantiation because shuffle() is not called during training (see set_seed)

        data_extractor_train = aug.ShuffledDataExtractor(self.point_cloud_shuffler, categories)
        data_extractor_valid = aug.ShuffledDataExtractor(self.point_cloud_shuffler_valid, categories)
        data_extractor_test = aug.ShuffledDataExtractor(self.point_cloud_shuffler_valid,
                                                        categories=('indices', 'labels'))

        # define extractors
        self.extractor_train = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.SubjectExtractor(),  # required for plotting
             PointCloudSizeExtractor(),  # to init_shape in SubjectAssembler
             data_extractor_train,
             pymia_extr.IndexingExtractor(),  # for SubjectAssembler (assembling)
             pymia_extr.ImageShapeExtractor()  # for SubjectAssembler (shape)
             ])

        self.extractor_valid = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.SubjectExtractor(),  # required for plotting
             PointCloudSizeExtractor(),  # to init_shape in SubjectAssembler
             data_extractor_valid,
             pymia_extr.IndexingExtractor(),
             pymia_extr.ImageShapeExtractor()
             ])

        self.extractor_test = pymia_extr.ComposeExtractor(
            [pymia_extr.NamesExtractor(),  # required for SelectiveDataExtractor
             pymia_extr.SubjectExtractor(),
             data_extractor_test,  # we need the indices, i.e. the point's coordinates,
             # to convert the point cloud back to an image
             pymia_extr.DataExtractor(categories=('gt',), ignore_indexing=True),  # the ground truth is used for the
             # validation at config.save_validation_nth_epoch
             pymia_extr.ImagePropertiesExtractor(),
             pymia_extr.ImageShapeExtractor()
             ])

        # define transforms for extraction
        self.extraction_transform_train = pymia_tfm.ComposeTransform(
            [pymia_tfm.Squeeze(entries=('labels',), squeeze_axis=-1),  # for PyTorch loss functions
             pymia_tfm.LambdaTransform(lambda_fn=lambda np_data: np_data.astype(np.int64), entries=('labels', )),
             # for PyTorch loss functions
             ])

        self.extraction_transform_valid = pymia_tfm.ComposeTransform(
            [pymia_tfm.Squeeze(entries=('labels',), squeeze_axis=-1),  # for PyTorch loss functions
             pymia_tfm.LambdaTransform(lambda_fn=lambda np_data: np_data.astype(np.int64), entries=('labels', ))
             # for PyTorch loss functions
             ])

        if config.use_jitter:
            self.extraction_transform_train.transforms.append(aug.PointCloudJitter())
            self.extraction_transform_valid.transforms.append(aug.PointCloudJitter())

        if config.use_rotation:
            self.extraction_transform_train.transforms.append(aug.PointCloudRotate())
            self.extraction_transform_valid.transforms.append(aug.PointCloudRotate())

        # need to add probability concatenation after augmentation!
        if config.use_point_feature:
            self.extraction_transform_train.transforms.append(ConcatenateCoordinatesAndPointFeatures())
            self.extraction_transform_valid.transforms.append(ConcatenateCoordinatesAndPointFeatures())

        if config.use_image_information:
            spatial_size = config.image_information_config.spatial_size

            def slice_patches(np_data):
                z = (np_data.shape[1] - spatial_size) // 2
                y = (np_data.shape[2] - spatial_size) // 2
                x = (np_data.shape[3] - spatial_size) // 2

                np_data = np_data[:, z:(z + spatial_size), y:(y + spatial_size), x:(x + spatial_size), :]
                return np_data

            self.extraction_transform_train.transforms.append(pymia_tfm.LambdaTransform(
                lambda_fn=slice_patches, entries=image_information_categories))
            self.extraction_transform_valid.transforms.append(pymia_tfm.LambdaTransform(
                lambda_fn=slice_patches, entries=image_information_categories))

        # define loaders
        training_sampler = pymia_extr.SubsetRandomSampler(sampler_ids_train)
        self.loader_train = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_training,
                                                  sampler=training_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        validation_sampler = pymia_extr.SubsetSequentialSampler(sampler_ids_valid)
        self.loader_valid = pymia_extr.DataLoader(self.dataset,
                                                  config.batch_size_testing,
                                                  sampler=validation_sampler,
                                                  collate_fn=collate_fn,
                                                  num_workers=1)

        self.loader_test = pymia_extr.DataLoader(self.dataset,
                                                 config.batch_size_testing,
                                                 sampler=validation_sampler,
                                                 collate_fn=collate_fn,
                                                 num_workers=1)

        self.extraction_transform_test = None
