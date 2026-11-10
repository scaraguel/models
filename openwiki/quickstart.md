---
type: repository quickstart
title: Repository Quickstart
description: A task-oriented entry point to ml4t-models installation boundaries, stable imports, finance-native runtime domains, and the shortest documentation path for making and validating changes.
tags: [quickstart, installation, public-api, model-families, development]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-10T03:14:30.813Z
sources:
  - id: openwiki-source-8037e2358a2c4f9b2c722a11
    resource: repo://AGENTS.md
  - id: openwiki-source-879c710378f4939b0b8f9110
    resource: repo://docs/getting-started/quickstart.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-6fdaa492c5fe83db6cac0346
    resource: repo://src/ml4t/models/__init__.py
  - id: openwiki-source-9a7f01147166033252c87da0
    resource: repo://src/ml4t/models/_public_api.py
  - id: openwiki-source-6a40916908450fd68acf6c59
    resource: repo://src/ml4t/models/api.py
  - id: openwiki-source-65dc1e88be0384d48a13d231
    resource: repo://src/ml4t/models/pipelines.py
  - id: openwiki-source-48cff8bf4208ea16e730c6b1
    resource: repo://src/ml4t/models/types.py
  - id: openwiki-source-02b25b9dd71b5725b07e9e88
    resource: repo://tests/test_core_install.py
  - id: openwiki-source-e57c0cc30106f6c130e549d1
    resource: repo://tests/test_public_api.py
generated: { by: "openwiki/0.9.0", at: "2026-11-10T03:14:30.813Z" }
---

# Repository Quickstart

`ml4t-models` is the modeling layer of the ML4T ecosystem: it accepts identity-preserving financial batches, estimates structural state or direct predictions, and emits forecasts, signals, or target weights for downstream diagnostics and backtests. Its main domains are latent-factor estimation and forecasting, stochastic discount factor (SDF) estimation, direct asset prediction, and end-to-end portfolio learning. Start with the [System Map](architecture/system-map.md) when a change crosses more than one domain.

## Choose the installation boundary first

| Mode | Command | Intended boundary |
|---|---|---|
| Core | `pip install ml4t-models` | NumPy contracts and workflows, including the PCA quickstart and the linear portfolio baseline. No credentials, data download, accelerator, Torch, Polars, or `ml4t-specs` are required. |
| Deep | `pip install ml4t-models[deep]` | Adds PyTorch for CAE, SDF, SAE, LSTM portfolio, and deep portfolio behavior. |
| Integration | `pip install ml4t-models[integration]` | Adds Polars and `ml4t-specs` for long-frame/schema adapters, Parquet surfaces, and sibling-library handoffs. It does not imply Torch. |
| Complete runtime | `pip install ml4t-models[all]` | Installs both `deep` and `integration`. |
| Contributor checkout | `uv sync --locked --dev --extra docs` | Installs the locked development and documentation environment used by repository gates. |

The Python and Torch support boundaries are deliberately different. All model families are qualified on Python 3.12, 3.13, and 3.14. The package permits Python 3.15 for NumPy-based core workflows, but the `deep` extra declares Torch only when `python_version < '3.15'`; use Python 3.12–3.14 for neural models. Do not make a core import depend eagerly on Torch or an integration package.

## Import from the stable root

Consumer code must import supported symbols from `ml4t.models`, not implementation modules:

```python
from ml4t.models import CrossSectionBatch, IPCAConfig, IPCAModel
from ml4t.models import ExpandingMeanFactorForecaster, LatentFactorForecastPipeline
```

The root API covers the finance-native batches and results, model protocols, configurations, public model classes, pipelines, forecasters, mappers, and integration helpers. `src/ml4t/models/_public_api.py` is the authoritative stable-export manifest, and `tests/test_public_api.py` requires it to match `ml4t.models.__all__`. Internal bases, losses, and helpers are intentionally not part of that contract. Several portfolio classes are loaded lazily; if PyTorch is absent, requesting a neural class raises an `ImportError` that points to `ml4t-models[deep]`, while the core linear portfolio workflow remains usable.

For any contract or export change, read [Finance-Native Contracts and Public API](concepts/contracts-and-public-api.md) before changing call sites. Adding a symbol to an internal module does not make it public.

## Smallest complete core workflow

This CPU-only path makes the repository’s separation of concerns concrete: PCA extracts structural factor state, a forecaster estimates ex ante factor premia from fitted factor history, and a mapper turns state plus premia into asset expected returns.

```python
import numpy as np

from ml4t.models import (
    BetaLambdaMapper,
    ExpandingMeanFactorForecaster,
    LatentFactorForecastPipeline,
    PCAConfig,
    PCAModel,
    PersistentPanelBatch,
)

rng = np.random.default_rng(1)
asset_ids = tuple(f"asset_{i}" for i in range(6))
train = PersistentPanelBatch(
    returns=rng.normal(scale=0.02, size=(12, 6)),
    timestamps=tuple(f"2024-{month:02d}" for month in range(1, 13)),
    asset_ids=asset_ids,
)
future = PersistentPanelBatch(
    timestamps=("2025-01", "2025-02"),
    asset_ids=asset_ids,
)

pipeline = LatentFactorForecastPipeline(
    model=PCAModel(PCAConfig(n_factors=2)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)
pipeline.fit(train)
forecast = pipeline.predict(future).asset_forecast.expected_returns
assert forecast.shape == (2, 6)
assert np.isfinite(forecast).all()
```

`LatentFactorForecastPipeline.fit()` fits the structural model, extracts training state, then fits the forecaster. `predict()` extracts state for the requested batch and checkpoint, forecasts factor premia, and asks the mapper for asset forecasts. The future batch therefore supplies timestamps and persistent asset identity, not future returns. The result keeps rows in future timestamp order and columns in `asset_ids` order. These are synthetic model forecasts, not evidence of trading performance. See the repository’s [full getting-started quickstart](../docs/getting-started/quickstart.md) for the runnable script and expected output.

## Route a change by the product it owns

Do not begin from a similarly named class. First decide whether the change concerns a contract, structural state, factor forecast, direct signal, SDF weight, target allocation, runtime service, or downstream handoff.

| Change or question | Shortest route |
|---|---|
| Package ownership or a change crossing domains | [System Map and Ownership Boundaries](architecture/system-map.md) |
| Batch axes, masks, identifiers, protocols, result objects, or a root export | [Finance-Native Contracts and Public API](concepts/contracts-and-public-api.md) |
| Choosing among PCA, RP-PCA, IPCA, CAE, direct SAE prediction, SDF, or portfolio learners | [Model Families and Selection Boundaries](architecture/model-families.md) |
| Structural extraction followed by factor-premium forecasting and beta–lambda mapping | [Latent-Factor Forecasting Workflow](workflows/latent-factor-forecasting.md) |
| Unconditional, adversarial, or conditional SDF phases; SDF checkpoints; weight extraction; optional return mapping | [Stochastic Discount Factor Training and Extraction](workflows/stochastic-discount-factor.md) |
| Linear, LSTM, or deep allocation; objectives; validation; postprocessing; raw versus final weights | [Portfolio Learning and Allocation Workflow](workflows/portfolio-learning.md) |
| Device/dtype resolution, seeds, fit rollback, observability, checkpoints, save/load, or artifact compatibility | [Training Runtime, Observability, and Artifacts](operations/training-runtime-and-artifacts.md) |
| Long-frame input, schema resolution, prediction/signal/weight frames, Parquet publication, or `ml4t-backtest` handoff | [Data Adapters and Downstream Handoffs](integrations/data-and-backtest-handoffs.md) |
| Focused tests, financial invariants, optional-dependency checks, full quality gates, or release qualification | [Change Validation and Test Strategy](testing/change-validation.md) |

Direct SAE work belongs to the direct-prediction family: it consumes `CrossSectionBatch` and produces `AssetSignalResult`; it does not enter the latent-factor forecaster/mapper path. Likewise, SDF estimation is weight-native, and portfolio models produce allocations directly. Do not force either through a beta-times-lambda abstraction.

## Invariants to preserve while changing code

1. **Preserve financial identity.** Time, asset, batch, and feature axes—and their timestamps, asset IDs, masks, or groups—are part of the contract, not incidental metadata.
2. **Keep stages explicit.** Structural extraction, factor-premium forecasting, asset mapping, direct prediction, and allocation have different protocols and result types.
3. **Keep optional dependencies at their boundary.** Core workflows must remain importable without Torch, Polars, or `ml4t-specs`; fail with an actionable extra only when optional behavior is requested.
4. **Treat lifecycle state as observable.** Fitted state, checkpoints, fit records, rollback behavior, and artifact compatibility are user-visible behavior, especially for stochastic training.
5. **Hand off rather than duplicate.** This package prepares model results and feed inputs; execution simulation belongs to `ml4t-backtest`, while reusable evaluation belongs to `ml4t-diagnostic`.
6. **Test economics as well as shape.** Identity preservation, no-lookahead behavior, normalization, turnover/cost behavior, checkpoint selection, and persistence failure semantics need focused coverage where applicable.

## Validate from narrow to complete

Use the focused suite listed in [Change Validation and Test Strategy](testing/change-validation.md) while iterating. Before presenting a complete repository change, run every applicable repository gate:

```bash
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check
uv run pytest tests/ -q --cov-report=json:coverage.json
uv run python scripts/ci/check_coverage.py coverage.json
uv run mkdocs build --strict
uv build
```

A passing shape test is not enough when a change can alter financial alignment, stochastic lifecycle behavior, an optional-install boundary, artifact safety, or a downstream handoff.
