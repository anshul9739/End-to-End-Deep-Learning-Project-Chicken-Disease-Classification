from pathlib import Path
from typing import Any
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = Path("config/config.yaml"),
        params_filepath: Path = Path("params.yaml"),
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        create_directories([cfg["root_dir"]])
        return DataIngestionConfig(
            root_dir=Path(cfg["root_dir"]),
            source_URL=cfg["source_URL"],
            local_data_file=Path(cfg["local_data_file"]),
            unzip_dir=Path(cfg["unzip_dir"]),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        cfg = self.config["prepare_base_model"]
        P = self.params
        create_directories([cfg["root_dir"]])
        return PrepareBaseModelConfig(
            root_dir=Path(cfg["root_dir"]),
            base_model_path=Path(cfg["base_model_path"]),
            updated_base_model_path=Path(cfg["updated_base_model_path"]),
            params_image_size=P["IMAGE_SIZE"],
            params_learning_rate=P["LEARNING_RATE"],
            params_include_top=P["INCLUDE_TOP"],
            params_weights=P["WEIGHTS"],
            params_classes=P["CLASSES"],
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        cfg = self.config["prepare_callbacks"]
        create_directories(
            [
                cfg["root_dir"],
                cfg["tensorboard_root_log_dir"],
                Path(cfg["checkpoint_model_filepath"]).parent,
            ]
        )
        return PrepareCallbacksConfig(
            root_dir=Path(cfg["root_dir"]),
            tensorboard_root_log_dir=Path(cfg["tensorboard_root_log_dir"]),
            checkpoint_model_filepath=Path(cfg["checkpoint_model_filepath"]),
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        params = self.params
        data_root = Path(self.config["data_ingestion"]["unzip_dir"]) / "Chicken-fecal-images"
        create_directories([cfg["root_dir"]])
        return TrainingConfig(
            root_dir=Path(cfg["root_dir"]),
            trained_model_path=Path(cfg["trained_model_path"]),
            training_data=data_root,
            params_batch_size=params["BATCH_SIZE"],
            params_epochs=params["EPOCHS"],
            params_is_augmentation=params["AUGMENTATION"],
            params_image_size=params["IMAGE_SIZE"],
        )

    def get_validation_config(self) -> EvaluationConfig:
        """Return config for the evaluation/validation stage."""
        params = self.params
        data_root = Path(self.config["data_ingestion"]["unzip_dir"]) / "Chicken-fecal-images"
        return EvaluationConfig(
            path_of_model=Path(self.config["training"]["trained_model_path"]),
            training_data=data_root,
            all_params=params,
            params_image_size=params["IMAGE_SIZE"],
            params_batch_size=params["BATCH_SIZE"],
        )

