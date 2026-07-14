"""Typed, configuration-complete cache keys and storage for attack artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Sequence

from torch.utils.data import Subset


def _freeze(value: Any) -> Hashable:
    """Convert nested attack parameters to a deterministic hashable value."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    return repr(value)


def loader_dataset_indices(loader) -> tuple[int, ...]:
    """Return the underlying dataset indices represented by a data loader."""
    dataset = loader.dataset
    indices: Sequence[int] = range(len(dataset))
    while isinstance(dataset, Subset):
        parent_indices = dataset.indices
        indices = tuple(int(parent_indices[index]) for index in indices)
        dataset = dataset.dataset
    return tuple(int(index) for index in indices)


@dataclass(frozen=True)
class AttackKey:
    """Identity of an attack computation whose result can be reused safely."""

    model_id: int | str
    attack_name: str
    epsilon: float | None = None
    alpha: float | None = None
    steps: int | None = None
    seeds: tuple[int, ...] = ()
    restarts: int = 1
    use_ste: bool = False
    dataset_indices: tuple[int, ...] = ()
    attack_parameters: tuple[tuple[str, Hashable], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        model_id: int | str,
        attack_name: str,
        loader,
        epsilon: float | None = None,
        alpha: float | None = None,
        steps: int | None = None,
        seeds: Sequence[int] = (),
        restarts: int = 1,
        use_ste: bool = False,
        attack_parameters: Mapping[str, Any] | None = None,
    ) -> "AttackKey":
        parameters = {
            "batch_size": getattr(loader, "batch_size", None),
            "drop_last": getattr(loader, "drop_last", False),
            "dataset_type": type(loader.dataset).__qualname__,
            **(attack_parameters or {}),
        }
        return cls(
            model_id=model_id,
            attack_name=str(attack_name),
            epsilon=None if epsilon is None else float(epsilon),
            alpha=None if alpha is None else float(alpha),
            steps=None if steps is None else int(steps),
            seeds=tuple(int(seed) for seed in seeds),
            restarts=int(restarts),
            use_ste=bool(use_ste),
            dataset_indices=loader_dataset_indices(loader),
            attack_parameters=tuple(
                sorted((str(name), _freeze(value)) for name, value in parameters.items())
            ),
        )


@dataclass
class AttackResultCache:
    """In-memory cache for attack results or generated adversarial artifacts."""

    _values: dict[AttackKey, Any] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_compute(self, key: AttackKey, compute: Callable[[], Any]) -> Any:
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        value = compute()
        self._values[key] = value
        return value

    def get(self, key: AttackKey, default=None):
        if key in self._values:
            self.hits += 1
            return self._values[key]
        self.misses += 1
        return default

    def store(self, key: AttackKey, value: Any) -> Any:
        self._values[key] = value
        return value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)
