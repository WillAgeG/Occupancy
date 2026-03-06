from models.lmsc.lmscnet import LMSCNet
from models.lmsc.lmscnet_2d import LMSCNet2d
from src.common import base_config, enums
from src.models.rwkv.occrwkv import OccRWKV, OccRWKV2D


def get_model(cfg: base_config.ExperimentConfig, dataset):

    nbr_classes = cfg.dataset.nbr_classes
    input_height = cfg.dataset.H
    class_frequencies = dataset.class_frequencies
    phase = dataset.phase

    selected_model = cfg.trainer.model_type

    # LMSCNet ----------------------------------------------------------------------------------------------------------
    if selected_model == enums.ModelType.LMSCNET:
        model = LMSCNet(class_num=nbr_classes, input_height=input_height,
                        class_frequencies=class_frequencies, phase=phase)
    # ------------------------------------------------------------------------------------------------------------------

    # LMSCNet2d --------------------------------------------------------------------------------------------------------
    elif selected_model == enums.ModelType.LMSCNET_2D:
        model = LMSCNet2d(class_num=nbr_classes, input_height=input_height,
                          class_frequencies=class_frequencies, phase=phase)
    # ------------------------------------------------------------------------------------------------------------------

    # OccRWKV2D --------------------------------------------------------------------------------------------------------
    elif selected_model == enums.ModelType.OCCRWKV_2D:
        model = OccRWKV2D(cfg, phase=phase)
    # ------------------------------------------------------------------------------------------------------------------

    # OccRWKV ----------------------------------------------------------------------------------------------------------
    elif selected_model == enums.ModelType.OCCRWKV:
        model = OccRWKV(cfg, phase=phase)
    # ------------------------------------------------------------------------------------------------------------------

    else:
        err = "Wrong model selected"
        raise AssertionError(err)

    return model
