"""
Plot the run results.
Your current working directory should contain the statistics files.

Make sure to change `PLOT_ONLY_BASIC_LINEAR_RULE` if you like. Also consider changing
    `dqn.runs.MAX_STEPS` to 5 million if you do so.
"""
import os

from dqn.runs import runs, t_array
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

PLOT_ONLY_BASIC_LINEAR_RULE = False  # change that if you want


def save_plot(results, attribute, ylim, name_prefix=''):
    font = FontProperties()
    font.set_size('small')

    ax = plt.subplot()

    handles = []
    for result in results:
        plot, = plt.plot(t_array, getattr(result, attribute), label=result.run_name)
        handles.append(plot)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width, box.height * 0.9])

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xlabel('t')
    plt.ylabel(attribute)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), prop=font)
    plt.grid()
    plt.xlim(0, t_array.max())
    if ylim is not None:
        plt.ylim(ylim)
    os.makedirs('../plots', exist_ok=True)
    plt.savefig('../plots/{}{}.png'.format(name_prefix, attribute))
    plt.clf()


if PLOT_ONLY_BASIC_LINEAR_RULE:
    save_plot([runs[0]], 'mean_episode_rewards', ylim=None, name_prefix='basic_linear_rule-')
    save_plot([runs[0]], 'best_mean_episode_rewards', ylim=None, name_prefix='basic_linear_rule-')
else:
    save_plot(runs, 'mean_episode_rewards', ylim=None)
    save_plot(runs, 'best_mean_episode_rewards', ylim=None)
    save_plot(runs, 'exploration_by_t', ylim=(0, 1.01))
