#-*-conding:utf-8-*-
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from .utils import *

def get_dist(tensor, axis=0, distribution_path="distribution.png"):
    '''
    Plot tensor distribution for tensor comprehension.
   
    :param tensor: input ndarray data.
    :return: distribution figure
    '''
    tensor = single_trans(tensor, )
    t_std = tensor_maxes_std(tensor)
    t_ratio = tensor_maxes_ratio(tensor)

    new_tensor = tensor.reshape([tensor.shape[0], -1])
    new_tensor = pd.DataFrame(np.abs(new_tensor)).T

    sns.set(style="ticks", color_codes=True, rc={'figure.figsize':(100, 80)}, font_scale = 1)
    g = sns.catplot(kind="strip", data=new_tensor, height=8, aspect=2, margin_titles=True)
    g.set_axis_labels("Channel Id", "Value")

    metric_to_color = {
        'Tensor maxes std = ' + str(t_std):   'white',
        'Tensor maxes ratio = ' + str(t_ratio):   'white',
    }
    patches = [matplotlib.patches.Patch(color=v, label=k) for k,v in metric_to_color.items()]
    plt.legend(handles=patches)

    figure = g.fig
    figure.savefig(distribution_path, dpi=400)
