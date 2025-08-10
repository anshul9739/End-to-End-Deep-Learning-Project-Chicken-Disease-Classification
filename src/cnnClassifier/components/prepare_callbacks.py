from __future__ import annotations
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig) -> None:
        self.config = config

    def _create_tb_callbacks(self):
        log_dir = str(self.config.tensorboard_root_log_dir)
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    def _create_ckpt_callbacks(self):
        ckpt_path = str(self.config.checkpoint_model_filepath)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

    def get_tb_ckpt_callbacks(self):
        return [self._create_tb_callbacks(), self._create_ckpt_callbacks()]
