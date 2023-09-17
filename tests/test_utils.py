import os
from laser_jitter.utils import (fix_seed, read_yaml, write_yaml)

yaml_file = 'test.yml'
data = {'a': 1,
        'b': 'hello'}


def test_fix_seed():
    fix_seed(101)


def test_write_yaml():
    write_yaml(yaml_file, data)


def test_read_yaml():
    data_ = read_yaml(yaml_file)
    assert data_ == data
    os.remove(yaml_file)
    