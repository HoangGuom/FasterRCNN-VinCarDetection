from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def make_anchor_generator(config: dict, levels: int = 5) -> AnchorGenerator:
    sizes = [int(value) for value in config["model_params"]["anchor_sizes"]]
    ratios = tuple(float(value) for value in config["model_params"]["anchor_ratios"])
    if len(sizes) < levels:
        sizes.extend([sizes[-1]] * (levels - len(sizes)))
    return AnchorGenerator(
        tuple((size,) for size in sizes[:levels]),
        (ratios,) * levels,
    )


def build_model(config: dict, pretrained: bool = False) -> FasterRCNN:
    """Tạo Faster R-CNN bản cuối: ConvNeXt-T backbone và FPN."""
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    backbone_body = convnext_tiny(weights=weights).features
    backbone = BackboneWithFPN(
        backbone_body,
        return_layers={"1": "0", "3": "1", "5": "2", "7": "3"},
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
    )

    model_cfg = config["model_params"]
    input_cfg = model_cfg["input"]
    rpn_cfg = model_cfg["rpn"]
    roi_cfg = model_cfg["roi"]
    return FasterRCNN(
        backbone,
        num_classes=int(config["dataset_params"]["num_classes"]),
        min_size=int(input_cfg["train_size"]),
        max_size=int(input_cfg["max_size"]),
        rpn_anchor_generator=make_anchor_generator(config),
        rpn_pre_nms_top_n_train=int(rpn_cfg["train_pre_nms_topk"]),
        rpn_post_nms_top_n_train=int(rpn_cfg["train_post_nms_topk"]),
        rpn_pre_nms_top_n_test=int(rpn_cfg["test_pre_nms_topk"]),
        rpn_post_nms_top_n_test=int(rpn_cfg["test_post_nms_topk"]),
        rpn_nms_thresh=float(rpn_cfg["nms_threshold"]),
        box_score_thresh=float(model_cfg["thresholds"]["eval_score"]),
        box_nms_thresh=float(roi_cfg["nms_threshold"]),
        box_detections_per_img=int(roi_cfg["detections_per_image"]),
    )
