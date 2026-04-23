"""
Batch inference pipeline: load registered model bundle and score new orders.

Usage:
    python pipelines/inference_pipeline.py
"""

from zenml import pipeline

from steps.inference import run_inference


@pipeline(name="supply_chain_inference_pipeline", enable_cache=False)
def inference_pipeline(
    input_path: str = "data/DataCoSupplyChainDataset.csv",
    output_path: str = "data/predictions.csv",
    features_config_path: str = "configs/features_config.yaml",
    model_name: str = "supply-chain-late-delivery",
    model_alias: str = "staging",
) -> None:
    run_inference(
        input_path=input_path,
        output_path=output_path,
        features_config_path=features_config_path,
        model_name=model_name,
        model_alias=model_alias,
    )


if __name__ == "__main__":
    inference_pipeline()
