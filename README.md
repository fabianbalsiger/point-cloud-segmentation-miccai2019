# Learning Shape Representation on Sparse Point Clouds for Volumetric Image Segmentation
This repository contains code for the [MICCAI 2019](https://www.miccai2019.org) paper "Learning Shape Representation on Sparse Point Clouds for Volumetric Image Segmentation", which can be found at [https://doi.org/10.1007/978-3-030-32245-8_31](https://doi.org/10.1007/978-3-030-32245-8_31).


## Installation

The installation has been tested with Ubuntu 16.04, Python 3.6, TensorFlow 1.10, and CUDA 9.0. The ``setup.py`` file lists all other dependencies.

First, create a virtual environment named `pc` with Python 3.6:

        $ virtualenv --python=python3.6 pc
        $ source ./pc/bin/activate

Second, copy the code:

        $ git clone https://github.com/fabianbalsiger/point-cloud-segmentation-miccai2019
        $ cd point-cloud-segmentation-miccai219

Third, install the required libraries:

        $ pip install -r requirements.txt

Fourth, compile the required TensorFlow extension for farthest point sampling. Note that you might need to edit ``compile.sh`` such that the compilation works.

        $ cd pc/model/sampling
        $ ./compile.sh

This should install all required dependencies. Please refer to the official TensorFlow documentation on how to use TensorFlow with [GPU support](https://www.tensorflow.org/install/gpu).

## Usage

We shortly describe the training procedure.
The data used in the paper is not publicly available. But, we provide a script to generate dummy data such that you are able to run the code.

### Dummy Data Generation and Configuration

We handle the data using [pymia](https://pymia.readthedocs.io/en/latest). Therefore, we need to create a hierarchical data format (HDF) file to have easy and fast access to our data during the training and testing.
Create the dummy data by

        $ python ./snippets/create_dataset.py

This will create the file ``./data/data.h5``, or simply our dataset. Use any open source HDF viewer to inspect the file (e.g., [HDFView](https://www.hdfgroup.org/downloads/hdfview/)).
Please refer to the [pymia documentation](https://pymia.readthedocs.io/en/latest/examples.dataset.html) on how to create your own dataset. 

Now, we create a training/validation/testing split file by

        $ python ./snippets/create_split.py

This will create the file ``./data/split1_04-02-02.json``.

Finally, you need to adjust the paths in the configuration file ``./bin/config.json``:

 - ``database_file``
 - ``result_dir``
 - ``model_dir``
 - ``split_file``

### Training
To train the model, simply execute ``./bin/main.py``. The data and training parameters are provided by the ``./bin/config.json``, which you can adapt to your needs.
Note that you might want to specify the CUDA device by

        $ CUDA_VISIBLE_DEVICES=0 python ./bin/main.py

The script will automatically use the training subjects defined in ``./data/split1_04-02-02.json`` and evaluate the model's performance after each epoch on the validation subjects.
The validation will be saved under the path ``result_dir`` specified in the configuration file ``./bin/config.json``.
The trained model will be saved under the path ``model_dir`` specified in the configuration file ``./bin/config.json``.
Further, the script logs the training and validation progress for visualization using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard).
Start the TensorBoard to observe the training:

        $ tensorboard --logdir=<path to the model_dir>

## Support
We leave an explanation of the code as exercise ;-). But if you found a bug or have a specific question, please open an issue or a pull request.

## Citation

If you use this work, please cite

```
Balsiger, F., Soom, Y., Scheidegger, O., & Reyes, M. (2019). Learning Shape Representation on Sparse Point Clouds for Volumetric Image Segmentation. In D. Shen, T. Liu, T. M. Peters, L. H. Staib, C. Essert, S. Zhou, … A. Khan (Eds.), Medical Image Computing and Computer Assisted Intervention – MICCAI 2019 (pp. 273–281). https://doi.org/10.1007/978-3-030-32245-8_31
```

```
@inproceedings{Balsiger2019b,
address = {Cham},
author = {Balsiger, Fabian and Soom, Yannick and Scheidegger, Olivier and Reyes, Mauricio},
booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2019},
doi = {10.1007/978-3-030-32245-8_31},
editor = {Shen, Dinggang and Liu, Tianming and Peters, Terry M. and Staib, Lawrence H. and Essert, Caroline and Zhou, Sean and Yap, Pew-Thian and Khan, Ali},
pages = {273--281},
publisher = {Springer},
series = {Lecture Notes in Computer Science},
title = {{Learning Shape Representation on Sparse Point Clouds for Volumetric Image Segmentation}},
volume = {11765},
year = {2019}
}
```

## License

The code is published under the [MIT License](https://github.com/fabianbalsiger/point-cloud-segmentation-miccai2019/blob/master/LICENSE).
