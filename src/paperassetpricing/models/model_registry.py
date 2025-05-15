from typing import Type, Dict
from .base_model import BaseModel

_registry: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    """
    Decorator to register a BaseModel under a string key.
    """

    def _(cls: Type[BaseModel]):
        _registry[name] = cls
        return cls

    return _


def get_model(name: str) -> Type[BaseModel]:
    try:
        return _registry[name]
    except KeyError:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")


def list_models() -> Dict[str, Type[BaseModel]]:
    return dict(_registry)
