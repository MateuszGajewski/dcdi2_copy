from copy import deepcopy


def combine_with_defaults(config, defaults):
    res = deepcopy(defaults)
    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res
