import dataclasses
import enum
import re
import types
import typing
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, _UnionGenericAlias


def read_config_file(config_file_path: str) -> Dict[str, Any]:
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file_path, "r", encoding="utf8") as f:
        config_all = yaml.load(f, Loader=loader)

    return config_all


def save_config(config: dataclass, path: str) -> None:
    with open(path, 'w') as file:
        yaml.dump(dataclasses.asdict(config), file)


def check_if_configs_are_set(config_all: Dict[str, Any]) -> None:
    required_keys = ['training', 'model']
    for key in config_all.keys():
        if key not in required_keys:
            raise ValueError(f'{key} is not a valid config key! Only supported config keys are {required_keys}.')


def wrapped_hyperparams(*args, **kwargs):
    def wrapper(check_class):

        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):

            for name, value in kwargs.items():

                # getting the field type
                ft = typing.get_type_hints(check_class).get(name, None)

                # If the type is a union type, we will check if the corresponding value is a dict
                # If the value is a dict we have the dataclass type or else we just have a value
                # (We only specify two optional items). UniongenericAlias is type of optional
                if type(value) == dict and (type(ft) == types.UnionType or type(ft) == _UnionGenericAlias):
                    union_args = typing.get_args(ft)
                    # 2nd arg is the desired dataclass subtype
                    ft = union_args[1]

                if dataclasses.is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name] = obj
                o_init(self, *args, **kwargs)

        check_class.__init__ = __init__

        return check_class

    return wrapper(args[0]) if args else wrapper


class HPTypes(str, enum.Enum):
    CATEGORICAL = 'categorical'
    INTEGER = 'integer'
    FLOAT = 'float'


@dataclass
class TuningOptions:
    name: str
    type: HPTypes
    low: Optional = None
    high: Optional = None
    step: Optional = None
    choices: Optional = None
    log: bool = False
    automatically_set: Optional[bool] = None

    def __post_init__(self):
        # If it is automatically set we do not check the other options. It is mainly used for hyperparameter
        # that depend on other hyperparameter (e.g. in transformer the dim_feedforward parameter)
        if self.automatically_set:
            return

        if self.type == HPTypes.CATEGORICAL.value:
            if self.choices is None:
                raise ValueError(f'If type {self.type} was set than you need specify the field choices')
        elif self.type == HPTypes.INTEGER.value or self.type == HPTypes.FLOAT.value:
            if self.low is None or self.high is None:
                raise ValueError(f'If type {self.type} was set than you need specify the fields low and high')
            if self.log and self.step:
                raise ValueError(f'Log and step cannot be set simultaneously! Choose either one of them.')
