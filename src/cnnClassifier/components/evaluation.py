from __future__ import annotations
from pathlib import Path
import tensorflow as tf

from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.valid_generator = None
        self.score = None

    def _make_valid_gen(self):
        image_size = tuple(self.config.params_image_size[:2])  # (224, 224)
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,  # match training split
        )
        self.valid_generator = datagen.flow_from_directory(
            str(self.config.training_data),
            target_size=image_size,
            batch_size=self.config.params_batch_size,
            subset="validation",
            class_mode="sparse",   # IMPORTANT: match training
            shuffle=False,
        )

    def _load_model(self) -> tf.keras.Model:
        model_path = Path(self.config.path_of_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return tf.keras.models.load_model(str(model_path))

    def _ensure_compiled(self, model: tf.keras.Model) -> tf.keras.Model:
        if getattr(model, "optimizer", None) is None or getattr(model, "loss", None) != "sparse_categorical_crossentropy":
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
        return model

    def evaluation(self) -> None:
        self._make_valid_gen()
        model = self._load_model()
        model = self._ensure_compiled(model)

        loss, acc = model.evaluate(self.valid_generator)
        self.score = {"loss": float(loss), "accuracy": float(acc)}

    def save_score(self) -> None:
        """Persist the last evaluation metrics to artifacts/evaluation/metrics.json"""
        if self.score is None:
            raise ValueError("Call evaluation() before save_score().")
        out = Path("artifacts") / "evaluation" / "metrics.json"
        save_json(out, self.score)
