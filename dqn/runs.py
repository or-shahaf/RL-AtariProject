import functools
import pickle
import numpy
from utils import schedule


MAX_STEPS = 2000000
t_array = numpy.arange(start=0, stop=MAX_STEPS, step=1, dtype=numpy.int)


class Run(object):
    def __init__(self, run_name, statistics_file_name, schedule):
        self.run_name = run_name
        self.statistics_file_name = statistics_file_name
        self.schedule = schedule

    @property
    def mean_episode_rewards(self):
        return numpy.array(self._statistics_dict['mean_episode_rewards'],
                           dtype=numpy.float)[:MAX_STEPS]

    @property
    def best_mean_episode_rewards(self):
        return numpy.array(self._statistics_dict['best_mean_episode_rewards'],
                           dtype=numpy.float)[:MAX_STEPS]

    @property
    @functools.lru_cache()
    def _statistics_dict(self):
        with open(self.statistics_file_name, 'rb') as f:
            return pickle.load(f)

    @property
    def exploration_by_t(self):
        return [self.schedule.value(t) for t in t_array]

    def __hash__(self):
        return hash(self.run_name)


max_piece_timestamp = 1000002
runs = [
    Run("LinearSchedule(1000000, 0.1)",
        r"master-statistics.pkl",
        schedule.LinearSchedule(1000000, 0.1)),
    Run("ConstantSchedule(0.05)",
        r"const05.pkl",
        schedule.ConstantSchedule(0.05)),
    Run("InverseExponentialSchedule(0.1, 1000000)",
        r"inv-exp-1000000-0.1.pkl",
        schedule.InverseExponentialSchedule(0.1, 1000000)),
    Run("LinearSchedule(511003, 0.1)",
        r"greedier-linear-schedule.pkl",
        schedule.LinearSchedule(511003, 0.1)),
    Run("PiecewiseSchedule([(0, 1), (333334, 0.95), (666668, 0.15), (1000002, 0.1)], 0.1)",
        r"piecewise-1-.95-.15-.1.pkl",
        schedule.PiecewiseSchedule([
            (0, 1),
            (max_piece_timestamp // 3, 0.95),
            (2 * max_piece_timestamp // 3, 0.15),
            (max_piece_timestamp, 0.1)
        ], outside_value=0.1)),
    Run("LinearSchedule(1500000, 0.1)",
        r"linear-1500000-0.1.pkl",
        schedule.LinearSchedule(1500000, 0.1)),
    Run("PiecewiseSchedule([(0, 1), (1000000, 0.1)], 0)",
        "piecewise-1-.1-0.pkl",
        schedule.PiecewiseSchedule([(0, 1), (1000000, 0.1)], outside_value=0.00))
]
