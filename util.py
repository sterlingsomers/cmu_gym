

import collections

def deep_update(d, u):

    """

    stack overflow answer by bscan

        https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    :param d:
    :param u:
    :return:
    """
    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = deep_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}

    return d
