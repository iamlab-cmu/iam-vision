import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

import detectron2
from detectron2.utils.logger import setup_logger
from Vision_based_pose_estimation.datasets.data_utils import *
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, wget
# from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

def main():
    # Download an image from coco dataset
    # wget.download('http://images.cocodataset.org/val2017/000000439715.jpg', '/media/input.jpg')

    cfg_est = OmegaConf.load("cfgs/estimator.yaml")
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(cfg_est["model"]["detectron_pretrained_model_path"]))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

    if cfg_est["model"]["use_pretrained"]:
        test_im_filename = "/home/saumyas/Projects/IAM-Vision/Vision_based_pose_estimation/media/input.jpg"
        im = cv2.imread(test_im_filename)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_est["model"]["detectron_pretrained_model_path"])
    else:
        dataset_name = cfg_est["data"]["my_dataset_name"]
        class_name = "mug"
        for d in ["train", "val"]:
            DatasetCatalog.register("data_" + d, lambda d=d: eval(cfg_est["data"]["make_dict_func"])(cfg_est["data"]["my_dataset_path"] + d))
            MetadataCatalog.get("data_" + d).set(thing_classes=[class_name])
        metadata = MetadataCatalog.get("data_train")
        dataset_dicts = eval(cfg_est["data"]["make_dict_func"])(cfg_est["data"]["my_dataset_path"] + "val")

        test_im_filename = dataset_dicts[10]["file_name"]
        im = cv2.imread(test_im_filename)
        cfg.MODEL.WEIGHTS = cfg_est["model"]["my_model_path"]

        cfg.DATASETS.TRAIN = (dataset_name,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = cfg_est["data"]["num_workers"]
        cfg.SOLVER.IMS_PER_BATCH = cfg_est["train_params"]["ims_per_batch"]
        cfg.SOLVER.BASE_LR = cfg_est["train_params"]["base_lr"]
        cfg.SOLVER.MAX_ITER = cfg_est["train_params"]["max_iter"]
        cfg.SOLVER.STEPS = []        # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg_est["model"]["batch_size_per_image"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_est["model"]["num_classes"]

    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    # import ipdb; ipdb.set_trace()
    # print(outputs["proposals"])

    # We can use `Visualizer` to draw the predictions on the image.

    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    # out = v.overlay_instances()
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('Prediction', out.get_image()[:, :, ::-1])


    # out_gt = v.draw_dataset_dict(dataset_dicts[10])
    # cv2.imshow('Ground Truth', out_gt.get_image()[:, :, ::-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()