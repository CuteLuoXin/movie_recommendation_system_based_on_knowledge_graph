import os
from typing import List, Callable, Tuple
from utility.decorator import logger

# 记下ds文件夹的路径，确保其它py文件调用时读文件路径正确
ds_path = os.path.join(os.path.dirname(__file__), 'ds')


def _read_ml(relative_path: str, separator: str) -> List[Tuple[int, int, int, int]]:
    data = []
    with open(os.path.join(ds_path, relative_path), 'r') as f:
        for line in f.readlines():
            values = line.strip().split(separator)
            user_id, movie_id, rating, timestamp = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            data.append((user_id, movie_id, rating, timestamp))
    return data


def _read_ml1m() -> List[Tuple[int, int, int, int]]:
    return _read_ml('ml-1m/ratings.dat', '::')


@logger('开始读数据，', ('data_name', 'expect_length', 'expect_user', 'expect_item'))
def _load_data(read_data_fn: Callable[[], List[tuple]], expect_length: int, expect_user: int, expect_item: int,
               data_name: str) -> List[tuple]:
    data = read_data_fn()
    n_user, n_item = len(set(d[0] for d in data)), len(set(d[1] for d in data))
    assert len(data) == expect_length, data_name + ' length ' + str(len(data)) + ' != ' + str(expect_length)
    assert n_user == expect_user, data_name + ' user ' + str(n_user) + ' != ' + str(expect_user)
    assert n_item == expect_item, data_name + ' item ' + str(n_item) + ' != ' + str(expect_item)
    return data


def ml1m() -> List[Tuple[int, int, int, int]]:
    return _load_data(_read_ml1m, 1000209, 6040, 3706, 'ml1m')
