import typing

import numpy as np
import pymia.evaluation.evaluator as eval
import pymia.evaluation.metric as pymia_metric

import pc.configuration.config as cfg


def init_evaluator(write_to_console: bool = True, csv_file: str = None, calculate_distance_metrics: bool = False):
    evaluator = eval.Evaluator(EvaluatorAggregator())
    if write_to_console:
        evaluator.add_writer(eval.ConsoleEvaluatorWriter(5))
    if csv_file is not None:
        evaluator.add_writer(eval.CSVEvaluatorWriter(csv_file))
    if calculate_distance_metrics:
        evaluator.metrics = [pymia_metric.DiceCoefficient(),
                             pymia_metric.HausdorffDistance(),
                             pymia_metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),
                             pymia_metric.VolumeSimilarity()]
    else:
        evaluator.metrics = [pymia_metric.DiceCoefficient(), pymia_metric.VolumeSimilarity()]
    evaluator.add_label(1, cfg.FOREGROUND_NAME)
    return evaluator


class EvaluatorAggregator(eval.IEvaluatorWriter):

    def __init__(self):
        self.metrics = {}
        self.results = {}

    def clear(self):
        self.results = {}
        for metric in self.metrics.keys():
            self.results[metric] = {}

    def write(self, data: list):
        """Aggregates the evaluation results.

        Args:
            data (list of list): The evaluation data,
                e.g. [['PATIENT1', 'BACKGROUND', 0.90], ['PATIENT1', 'TUMOR', '0.62']]
        """
        for metric, metric_idx in self.metrics.items():
            for data_item in data:
                if not data_item[1] in self.results[metric]:
                    self.results[metric][data_item[1]] = []
                self.results[metric][data_item[1]].append(data_item[metric_idx])

    def write_header(self, header: list):
        self.metrics = {}
        for metric_idx, metric in enumerate(header[2:]):
            self.metrics[metric] = metric_idx + 2
        self.clear()  # init results dict


class AggregatedResult:

    def __init__(self, label: str, metric: str, mean: float, std: float):
        self.label = label
        self.metric = metric
        self.mean = mean
        self.std = std

    def __eq__(self, other):
        return self.label == other.label and self.metric == other.metric

    def __lt__(self, other):
        return self.label[0] > other.label[0] and self.metric[0] > other.metric[0]


class AggregatedResultWriter:

    def __init__(self, file_path: str = None, precision: int = 3):
        super().__init__()
        self.file_path = file_path
        self.precision = precision

    def write(self, data: typing.List[AggregatedResult]):
        header = ['LABEL', 'METRIC', 'RESULT']
        data = sorted(data)

        # we store the output data as list of list to nicely format the intends
        out_as_string = [header]
        for result in data:
            out_as_string.append([result.label,
                                  result.metric,
                                  '{0:.{2}f} ± {1:.{2}f}'.format(result.mean, result.std, self.precision)])

        # determine length of each column for output alignment
        lengths = np.array([list(map(len, row)) for row in out_as_string])
        lengths = lengths.max(0)
        lengths += (len(lengths) - 1) * [2] + [0, ]  # append two spaces except for last column

        # format for output alignment
        out = [['{0:<{1}}'.format(val, lengths[idx]) for idx, val in enumerate(line)] for line in out_as_string]

        to_print = '\n'.join(''.join(line) for line in out)
        print(to_print)

        if self.file_path is not None:
            with open(self.file_path, 'w+') as file:
                file.write(to_print)


def aggregate_results(evaluator: eval.Evaluator) -> typing.List[AggregatedResult]:
    for writer in evaluator.writers:
        if isinstance(writer, EvaluatorAggregator):
            results = []
            for metric in writer.metrics.keys():
                for label, values in writer.results[metric].items():
                    results.append(AggregatedResult(label, metric, float(np.mean(values)), float(np.std(values))))
            writer.clear()
            return results
