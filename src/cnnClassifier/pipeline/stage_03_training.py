from tensorflow.keras.models import load_model
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier.components.training import Training


STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self) -> None:
        config = ConfigurationManager()

        cb_cfg = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(cb_cfg)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        train_cfg = config.get_training_config()

        try:
            model = load_model("artifacts/prepare_base_model/base_model_updated.h5")
        except Exception:
            model = load_model("artifacts/prepare_base_model/base_model.h5")

        trainer = Training(train_cfg)
        trainer.train(model, callback_list)


if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.main()
