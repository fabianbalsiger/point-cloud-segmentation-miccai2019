import json
import os


def split_subject_by_amount(subjects: list, amount: tuple) -> tuple:
    if sum(amount) != len(subjects):
        raise ValueError('Can not split n={} subjects into the splits'.format(len(subjects)), amount)

    with_test = len(amount) == 3

    train_subjects = subjects[:amount[0]]
    valid_subjects = subjects[amount[0]:amount[0] + amount[1]]
    ret = [train_subjects, valid_subjects]

    if with_test:
        test_subjects = subjects[amount[0] + amount[1]:]
        ret.append(test_subjects)
    return tuple(ret)


def save_split(file_path: str, train_subjects: list, valid_subjects: list, test_subjects: list=None):
    if os.path.exists(file_path):
        os.remove(file_path)

    write_dict = {'train': train_subjects, 'valid': valid_subjects, 'test': test_subjects}

    with open(file_path, 'w') as f:
        json.dump(write_dict, f)


def load_split(file_path: str, k=None):
    with open(file_path, 'r') as f:
        read_dict = json.load(f)

    train_subjects, valid_subjects, test_subjects = read_dict['train'], read_dict['valid'], read_dict['test']
    if k is not None:
        train_subjects, valid_subjects = train_subjects[k], valid_subjects[k]

    if test_subjects is None or len(test_subjects) == 0:
        return train_subjects, valid_subjects
    else:
        return train_subjects, valid_subjects, test_subjects
