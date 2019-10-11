import os

import numpy as np
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import pymia.data.conversion as pymia_conv


class PointCloudPlotter:

    def __init__(self, path: str):
        self.path = path

    def plot(self, subject: str, prediction: np.ndarray, label: np.ndarray, coordinates: np.ndarray,
             properties: pymia_conv.ImageProperties):
        # columns
        # - input point cloud with proba
        # (- point cloud of segmentation)
        # - predicted point cloud
        # - gt point cloud

        figs = []
        axes_ranges = []
        titles = []

        figs.append(self.make_scatter_plot(coordinates, label))
        figs.append(self.make_scatter_plot(coordinates[prediction == 1], 0))
        figs.append(self.make_scatter_plot(coordinates[label == 1], 1))
        axes_ranges.append(properties.size)
        axes_ranges.append(properties.size)
        axes_ranges.append(properties.size)
        titles.append('{} Input'.format(subject))
        titles.append('{} Prediction'.format(subject))
        titles.append('{} Ground Truth'.format(subject))

        cols = 3
        rows = 1  # todo: n subjects

        # assemble figure together
        fig = tls.make_subplots(rows=rows, cols=cols, print_grid=False,
                                specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}] for _ in range(rows)])

        fig_idx = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                fig.append_trace(figs[fig_idx], row, col)
                fig_idx += 1

        # fig['layout'].update(title='BLA Title', showlegend=False)
        # fix axis' ranges
        for idx, (x, y, z) in enumerate(axes_ranges, 1):
            key = 'scene' if idx == 1 else 'scene{}'.format(idx)
            fig['layout'][key].update(go.layout.Scene(xaxis=dict(range=[1, x]),
                                                      yaxis=dict(range=[1, y]),
                                                      zaxis=dict(range=[1, z]),
                                        )) #camera=dict(eye=dict(x=0, y=0, z=2.15))
            #fig['layout'][key].update(go.Layout(title='dsf'))
            #fig['layout'][key].update(annotations=[dict(z=40, x=0, y=0, text='my title')])

        py.offline.plot(fig, filename=os.path.join(self.path, subject + '.html'), auto_open=False)

    @staticmethod
    def make_scatter_plot(coordinates: np.ndarray, values: np.ndarray):
        x, y, z = coordinates.transpose()
        trace = go.Scatter3d(x=x, y=y, z=z, mode='markers',
                             marker=dict(
                                 size=1,
                                 color=values,  # set color to an array/list of desired values
                                 colorscale='Viridis',  # choose a colorscale
                                 opacity=0.5
                             )
                             )
        return trace
