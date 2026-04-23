"""
Shadow deployment pipeline:
Compare production and candidate aliases on same batch input.

Usage:
    python pipelines/shadow_pipeline.py
"""

from zenml import pipeline

from steps.shadow_compare import shadow_compare


@pipeline(name="supply_chain_shadow_pipeline", enable_cache=False)
def shadow_pipeline(
    deployment_config_path: str = "configs/deployment_config.yaml",
) -> None:
    _ = shadow_compare(config_path=deployment_config_path)


if __name__ == "__main__":
    shadow_pipeline()
