"""
Rollback utility for MLflow alias-based deployment.

Usage:
    python scripts/rollback_production.py --model-name supply-chain-late-delivery --to-version 12
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient


def rollback_alias(model_name: str, to_version: str) -> None:
    client = MlflowClient()
    current_production = client.get_model_version_by_alias(model_name, "production")
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=to_version,
    )
    print("Rollback complete.")
    print(f"  model={model_name}")
    print(f"  previous_production_version={current_production.version}")
    print(f"  new_production_version={to_version}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback production alias to a target version.")
    parser.add_argument(
        "--model-name",
        default="supply-chain-late-delivery",
        help="Registered model name.",
    )
    parser.add_argument(
        "--to-version",
        required=True,
        help="Target model version to assign to production alias.",
    )
    args = parser.parse_args()
    rollback_alias(args.model_name, args.to_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
