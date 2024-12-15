from typing import Optional
from models import registry
from models.params import ConfigParams

_REGISTERED_PRIOR_CLS = {}
_REGISTERED_PRIOR_CFG = {}


def register_builder(key: str):
    """Decorates a builder.

    The builder should be a Callable (a class or a function).
    Args:
      key: A `str` of key to look up the builder.

    Returns:
      A callable for using as class decorator that registers the decorated class
      for creation from an instance of task_config_cls.
    """
    return registry.register(_REGISTERED_PRIOR_CLS, key)


def build(config: ConfigParams, name: Optional[str] = None, **kwargs):
    builder = registry.lookup(_REGISTERED_PRIOR_CLS, name or config.type)
    return builder(config=config, **kwargs)


def get_config(name: str):
    """Looks up the `Config` according to the `name`."""
    cfg_creater = registry.lookup(_REGISTERED_PRIOR_CFG, name)
    return cfg_creater()


def register_config(key: str):
    return registry.register(_REGISTERED_PRIOR_CFG, key)