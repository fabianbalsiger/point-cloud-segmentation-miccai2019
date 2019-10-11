import pymia.config.configuration as cfg
import pymia.deeplearning.config as dlcfg


# Some configuration variables, which most likely do not change.
# Therefore, they are not added to the Configuration class
NO_CLASSES = 2  # number of classes for the segmentation task

FOREGROUND_NAME = 'Nerve'


class ImageInformationConfiguration(cfg.ConfigurationBase):
    """Represents a image evidence specific configuration."""

    VERSION = 1
    TYPE = 'CUBE'

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        super().__init__()
        self.spatial_size = 5  # input size (3-D)
        self.loss_weight = 1.0  # loss weight for 'autoencoder'

        self.ch_in = 1  # number of image information channels
        self.no_features = 16  # number of features extracted from image information


class Configuration(dlcfg.DeepLearningConfiguration):
    """Represents a configuration."""

    VERSION = 1
    TYPE = 'MAIN'

    @classmethod
    def version(cls) -> int:
        """Gets the version number of the configuration.

        Returns:
            The version number.
        """
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        """Gets the type of the configuration.

        Returns:
            The type of the configuration.
        """
        return cls.TYPE

    def __init__(self):
        """Initializes a new instance of the Configuration class."""
        super().__init__()
        self.split_file = './bin/splits/somesplit.json'

        self.model = ''  # string identifying the model for the model factory
        self.experiment = ''  # describes the experiment (appears in output directory and therefore in TensorBoard)

        # training configuration
        self.learning_rate = 0.01  # the learning rate
        self.dropout_p = 0.2  # the dropout probability [0, 1]
        self.use_rotation = True  # whether to augment with rotation or not
        self.use_jitter = True  # whether to augment with jitter or not
        self.channel_factor = 2  # easy way to increase the number of channels in the network

        # network configuration
        self.no_points = 2048  # number of points in the cloud
        self.use_point_feature = False  # uses the probability feature (as additional point feature)
        self.use_image_information = True  # uses the image information

        self.image_information_config = ImageInformationConfiguration()


def load(path: str, config_cls):
    """Loads a configuration file.

    Args:
        path (str): The path to the configuration file.
        config_cls (class): The configuration class (not an instance).

    Returns:
        (config_cls): The configuration.
    """

    return cfg.load(path, config_cls)
