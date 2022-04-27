from typing import Callable, Dict, Iterable, Optional, Any
from abc import ABC, abstractmethod
from typing import cast

CLASS_NAME = str
CLASS_PROBABILITY = float


class ModelEngine(ABC):
    @abstractmethod
    def fit(self, X: Iterable, y: Iterable, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: Iterable) -> Dict[CLASS_NAME, CLASS_PROBABILITY]:
        pass

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        pass


class ModelsRegistry:
    """Models factory."""

    registry: Dict[str, ModelEngine] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register model."""

        def inner_wrapper(
            wrapped_class: ModelEngine,
        ) -> ModelEngine:
            if name in cls.registry:
                print(f"ModelEngine {name} already exists. " "It will be replaced")

            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_model(cls, name: str, cfg: Dict[str, Any]) -> "ModelEngine":
        """Create moel."""
        if name in cls.registry:
            return cast(Callable, cls.registry[name])(cfg)
        else:
            raise ValueError(f"There is no model: {name}, available: {cls.registry.keys()}")
