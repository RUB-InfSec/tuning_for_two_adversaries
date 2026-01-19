import sys

from aenum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @classmethod
    def values(cls):
        return list(cls._value2member_map_.keys())


class AttackType(ExtendedEnum):
    NONE = "None"
    TRANSFER = "transfer"
    QUERY = "query"


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def to_matrix(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_pareto_front(results, objectives, all_points=False):
    all_results = [r for r in results if all(m in r.metrics for m in objectives)]
    pareto_front = []
    for candidate in all_results:
        dominated = False
        c_obj = [candidate.metrics[m] for m in objectives]
        for other in all_results:
            if other == candidate:
                continue
            o_obj = [other.metrics[m] for m in objectives]
            if dominates(o_obj, c_obj):  # if some other dominates candidate
                dominated = True
                break
        if not dominated or all_points:
            pareto_front.append(candidate)
    return pareto_front


def find_key_paths(d, key_value, path=None):
    if path is None:
        path = []

    result = []

    for key, value in d.items():
        if isinstance(value, dict):
            if list(value.keys()) == [key_value]:  # Check if the dict has only 'grid_search'
                result.append(path + [key])
            else:
                result.extend(find_key_paths(value, key_value, path + [key]))

    return result


def get_nested_attr(obj, paths):
    results = []
    path_names = []

    for path in paths:
        path_string = ""
        temp_obj = obj
        for idx, attr in enumerate(path):
            temp_obj = getattr(temp_obj, attr, None)
            if temp_obj is None:
                break

            path_string += attr
            if idx < len(path) - 1:
                path_string += "."

        path_names.append(path_string)
        results.append(temp_obj)
    return path_names, results


@staticmethod
def is_debug_mode():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
