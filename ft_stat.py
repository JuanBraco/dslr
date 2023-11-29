import pandas as pd


def clean_list(args):
    return [x for x in args if x == x]


def ft_min_max(args):
    args.sort()
    return args[0], args[-1]


def ft_mean(args):
    """calculating mean"""
    if isinstance(args, (pd.DataFrame, pd.Series)):
        clean_args = [x for x in args if pd.notna(x)]
    else:
        clean_args = [x for x in args if x == x]
    if len(clean_args) == 0:
        return float('nan')
    return sum(clean_args) / len(clean_args)


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
    if isinstance(args, (pd.DataFrame, pd.Series)):
        clean_args = [x for x in args if pd.notna(x)]
    else:
        clean_args = [x for x in args if x == x]
    mean = ft_mean(clean_args)
    n = len(clean_args)
    if n <= 1:
        return float('nan')
    return sum((x - mean) ** 2 for x in clean_args) / (n - 1)


def standardization(args):
    if isinstance(args, pd.Series):
        mean_x = ft_mean(args)
        std_x = ft_var(args) ** 0.5
        return args.apply(lambda xi: (xi - mean_x) / std_x if pd.notna(xi) else float('nan'))
    else:
        # Handle the case where x is a list
        mean_x = ft_mean(args)
        std_x = ft_var(args) ** 0.5
        return [(xi - mean_x) / std_x if xi == xi else float('nan') for xi in args]
