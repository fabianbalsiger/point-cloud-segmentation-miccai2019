import argparse
import collections
import os
import random

import pymia.data.extraction as pymia_extr

import pc.data.split as split


def main(hdf_file: str, save_dir: str, seed: int):
    with pymia_extr.get_reader(hdf_file, True) as reader:
        subjects = reader.get_subjects()

    # we have 8 subjects
    subject_no = (4, 2, 2)
    file_prefix = 'split{}'.format(seed)

    os.makedirs(save_dir, exist_ok=True)
    random.Random(seed).shuffle(subjects)

    create_split(subjects, subject_no, save_dir, file_prefix=file_prefix)


def check_for_duplicates(subjects):
    duplicates = [item for item, count in collections.Counter(subjects).items() if count > 1]
    if len(duplicates) > 0:
        raise RuntimeError('Duplicates in train/valid/test split: ', duplicates)


def create_split(subjects, amount: tuple, save_dir: str, file_prefix: str = ''):
    train_subjects, valid_subjects, test_subjects = split.split_subject_by_amount(subjects, amount)

    check_for_duplicates(train_subjects + valid_subjects + test_subjects)

    split_str = '-'.join('{:02}'.format(int(i)) for i in amount)
    split_file = os.path.join(save_dir, '{}_{}.json'.format(file_prefix, split_str))
    split.save_split(split_file, train_subjects, valid_subjects, test_subjects)


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Train/valid/test split creation')

    parser.add_argument(
        '--hdf_file',
        type=str,
        default='../data/data.h5',
        help='Path to the dataset file.'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='../data',
        help='Path to the output directory.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='The seed to create the split.'
    )

    args = parser.parse_args()
    main(args.hdf_file, args.save_dir, args.seed)
