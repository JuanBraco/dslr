def clean_list(args):
    return [x for x in args if x == x]


def ft_min_max(args):
    args.sort()
    return args[0], args[-1]


def ft_mean(args):
    """calculating mean"""
    return sum(args) / len(args)


def ft_median(args):
    """calculating median"""
    sorted_args = sorted(args)
    n = len(sorted_args)
    if n % 2 == 1:
        return sorted_args[n // 2]
    else:
        return (sorted_args[n // 2 - 1] + sorted_args[n // 2]) / 2


def ft_quartile(args):
    """calculating first and third quartile using linear interpolation"""
    sorted_args = sorted(args)
    n = len(sorted_args)
    q1_idx = (n - 1) // 4
    q1_f = (n - 1) % 4 / 4
    q1 = (1 - q1_f) * sorted_args[q1_idx] + q1_f * sorted_args[q1_idx + 1]
    q3_idx = 3 * (n - 1) // 4
    q3_f = (3 * (n - 1)) % 4 / 4
    q3 = (1 - q3_f) * sorted_args[q3_idx] + q3_f * sorted_args[q3_idx + 1]
    return [q1, q3]


def ft_var(args):
    mean = ft_mean(args)
    return sum((x - mean) ** 2 for x in args) / (len(args) - 1)


def normalize(x):
    return ((x - x.min()) / (x.max() - x.min()))
