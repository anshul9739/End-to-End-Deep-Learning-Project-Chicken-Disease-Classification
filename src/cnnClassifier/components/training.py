from __future__ import annotations
from pathlib import Path
from typing import List
import tensorflow as tf

from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    """
    Handles data generators and the training loop.
    Expects integer labels from generators (class_mode='sparse') and therefore
    compiles the model with 'sparse_categorical_crossentropy'.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def _make_gens(self):
        image_size = tuple(self.config.params_image_size[:2])  # e.g. (224, 224)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            horizontal_flip=self.config.params_is_augmentation,
            rotation_range=20 if self.config.params_is_augmentation else 0,
            width_shift_range=0.1 if self.config.params_is_augmentation else 0.0,
            height_shift_range=0.1 if self.config.params_is_augmentation else 0.0,
            zoom_range=0.1 if self.config.params_is_augmentation else 0.0,
        )

        train_gen = datagen.flow_from_directory(
            str(self.config.training_data),
            target_size=image_size,
            batch_size=self.config.params_batch_size,
            subset="training",
            class_mode="sparse",
        )

        val_gen = datagen.flow_from_directory(
            str(self.config.training_data),
            target_size=image_size,
            batch_size=self.config.params_batch_size,
            subset="validation",
            class_mode="sparse",
        )

        return train_gen, val_gen

    def _ensure_compiled(self, model: tf.keras.Model) -> tf.keras.Model:
        needs_compile = (
            getattr(model, "optimizer", None) is None
            or getattr(model, "loss", None) != "sparse_categorical_crossentropy"
        )

        if needs_compile:
            model.compile(
                optimizer=(
                    model.optimizer
                    if getattr(model, "optimizer", None) is not None
                    else tf.keras.optimizers.SGD(learning_rate=0.01)
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        return model

    def train(
        self,
        model: tf.keras.Model,
        callback_list: List[tf.keras.callbacks.Callback],
    ) -> None:
        train_gen, val_gen = self._make_gens()
        model = self._ensure_compiled(model)

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config.params_epochs,
            callbacks=callback_list,
        )

        out_path = Path(self.config.trained_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out_path))
