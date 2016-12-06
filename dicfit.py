import scipy.optimize as opt
from collections import OrderedDict, Iterable


def _dict(keys, values):
    """Convert to a dictionary, with duplicate keys accumulated into a list.

    _dict(['A', 'A', 'B'], [0.0, 1.0, 2.0]) -> {'A': [0.0, 1.0], 'B': 2.0}
    """
    result = {}
    for key, value in zip(keys, values):
        if key in result:
            if not isinstance(result[key], list):
                result[key] = [result[key]]
            result[key].append(value)
        else:
            result[key] = value
    return result


def _wrapper(func):
    return lambda params, keys, fixed, data: func(*data, **fixed, **_dict(keys, params)).flatten()


def fix(dict_params, fixed_key):
    if '@'+fixed_key not in dict_params.keys():
        if fixed_key in dict_params.keys():
            if isinstance(dict_params, OrderedDict):
                list_params = []
                while dict_params:
                    list_params.append(dict_params.popitem(last=False))
                    if list_params[-1][0] == fixed_key:
                        list_params[-1] = ('@'+fixed_key, list_params[-1][1])
                for key, value in list_params:
                    dict_params[key] = value
            else:
                dict_params['@'+fixed_key] = dict_params.pop(fixed_key)
        else:
            raise Exception('"{0}" not found in dictionary!'.format(fixed_key))


def leastsq(func, dict_params, args=()):
    """
    A modified least square fitting routine that uses a dictionary as fitting parameters.
    Fitting parameters prepended with `@` are not optimized.
    """
    params = []
    keys = []
    fixed = {}
    for key, value in dict_params.items():
        if key[0] == '@':
            fixed[key[1:]] = value
        else:
            if isinstance(value, Iterable):
                params.extend(value)
                keys.extend([key]*len(value))
            else:
                params.append(value)
                keys.append(key)
    params, status = opt.leastsq(_wrapper(func), params, args=(keys, fixed, args))
    params = _dict(keys, params)
    if isinstance(dict_params, OrderedDict):
        result = OrderedDict(params)
        for key in dict_params.keys():
            result.move_to_end(key.strip('@'))
    else:
        result = {**fixed, **params}
    return result, status
