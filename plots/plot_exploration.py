import numpy
import torch
import matplotlib.pyplot as plt

import utils.schedule

t_array = numpy.arange(start=0, stop=2000000, step=100, dtype=numpy.int)


def get_explore_by_t(schedule):
    return [schedule.value(t) for t in t_array]


max_piece_timestamp = 1000002

linear_explore_by_t = get_explore_by_t(utils.schedule.LinearSchedule(1000000, 0.1))
linear_0_explore_by_t = get_explore_by_t(utils.schedule.PiecewiseSchedule(
    [(0, 1), (1000000, 0.1)], outside_value=0.00))
greedy_linear_explore_by_t = get_explore_by_t(utils.schedule.LinearSchedule(511003, 0.1))
nongreedy_linear_explore_by_t = get_explore_by_t(utils.schedule.LinearSchedule(1500000, 0.1))
inv_exp1_explore_by_t = get_explore_by_t(utils.schedule.InverseExponentialSchedule(0.1, 1000000))
const_explore_by_t = get_explore_by_t(utils.schedule.ConstantSchedule(0.05))
const1_explore_by_t = get_explore_by_t(utils.schedule.ConstantSchedule(0.1))
piece_explore_by_t = get_explore_by_t(utils.schedule.PiecewiseSchedule(
    [(0, 1), (max_piece_timestamp // 3, 0.95), (2 * max_piece_timestamp // 3, 0.15),
     (max_piece_timestamp, 0.1)], outside_value=0.1))

# up next: nongreed_line, linear_0

linear_0, = plt.plot(t_array, linear_0_explore_by_t, label="PiecewiseSchedule to 0")
line, = plt.plot(t_array, linear_explore_by_t, label="LinearSchedule(1000000, 0.1)")
greed_line, = plt.plot(t_array, greedy_linear_explore_by_t, label="LinearSchedule(511003, 0.1)")
nongreed_line, = plt.plot(t_array, nongreedy_linear_explore_by_t, label="LinearSchedule(1500000, "
                                                                        "0.1)")
inv_exp1, = plt.plot(t_array, inv_exp1_explore_by_t, label="InverseExponentialSchedule(0.1, 1000000)")
const, = plt.plot(t_array, const_explore_by_t, label="ConstantSchedule(0.05)")
const1, = plt.plot(t_array, const1_explore_by_t, label="ConstantSchedule(0.1)")
piece, = plt.plot(t_array, piece_explore_by_t, label="PiecewiseSchedule of thirds")
plt.xlabel('t')
plt.ylabel('exploration')
plt.legend(handles=[line, linear_0, greed_line, nongreed_line, inv_exp1, const, const1,
                    piece])
plt.grid()
plt.xlim(0, t_array.max())
plt.ylim(0, 1.01)
plt.savefig('./exploration.png')
