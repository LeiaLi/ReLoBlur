import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def build_predictor():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x'
    cfg.merge_from_file(model_zoo.get_config_file(f"{config_file}.yaml"))
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"{config_file}.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def get_segmentation_masks(image, predictor):
    with torch.no_grad():
        outputs = predictor(image)['instances']
    return outputs.pred_masks
