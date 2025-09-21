from enum import Enum
from typing import Annotated

import pytest
from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel

from pydantic_optuna_bridge import attach_optuna_metadata, derive_optuna_metadata


class Optimizer(str, Enum):
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SampleConfig(BaseModel):
    optimizer: Optimizer
    learning_rate: Annotated[float, Gt(1e-5), Lt(1.0)]
    hidden_units: Annotated[int, Ge(32), Le(256)]


@pytest.fixture
def metadata() -> dict[str, dict[str, object]]:
    return derive_optuna_metadata(
        SampleConfig,
        log_scale_fields={"learning_rate"},
        categorical_field_weights={"optimizer": [0.5, 0.3, 0.2]},
    )


def test_metadata_contains_expected_distributions(metadata: dict[str, dict[str, object]]) -> None:
    assert set(metadata) == {"optimizer", "learning_rate", "hidden_units"}

    assert metadata["optimizer"] == {
        "distribution": "categorical",
        "choices": ["adam", "sgd", "rmsprop"],
        "weights": [0.5, 0.3, 0.2],
    }

    assert metadata["learning_rate"] == {
        "distribution": "float",
        "low": pytest.approx(1e-5, rel=1e-9),
        "high": pytest.approx(1.0, rel=1e-9),
        "log": True,
    }

    assert metadata["hidden_units"] == {
        "distribution": "int",
        "low": 32,
        "high": 256,
    }


def test_attach_optuna_metadata_sets_json_schema(metadata: dict[str, dict[str, object]]) -> None:
    attach_optuna_metadata(SampleConfig, metadata)

    for field_name, field in SampleConfig.model_fields.items():
        assert field.json_schema_extra == {"optuna": metadata[field_name]}


def test_unknown_log_field_raises_assertion() -> None:
    with pytest.raises(AssertionError):
        derive_optuna_metadata(SampleConfig, log_scale_fields={"missing"})
