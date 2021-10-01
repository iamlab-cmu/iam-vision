import wget, random, cv2, os
from zipfile import ZipFile
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from Vision_based_pose_estimation.datasets.data_utils import *
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

def main():
    # Download custom dataset
    #TODO: replace below with loading custom dataset
    # wget.download('https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip', '../datasets/')
    # # Create a ZipFile Object and load sample.zip in it
    # with ZipFile('../datasets/balloon_dataset.zip', 'r') as zipObj:
    #    # Extract all the contents of zip file in current directory
    #    zipObj.extractall()

    class_name = "mug"
    dataset_path = "/home/saumyas/Projects/IAM-Vision/Vision_based_pose_estimation/Vision_based_pose_estimation/datasets/" + class_name + "/" #TODO: add cfg value here
    # Register the balloon dataset to detectron2
    for d in ["train", "val"]:
        DatasetCatalog.register("data_" + d, lambda d=d: xml_to_dict(dataset_path + d))
        MetadataCatalog.get("data_" + d).set(thing_classes=[class_name])
    balloon_metadata = MetadataCatalog.get("data_train")

    # To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:
    if False:
        dataset_dicts = get_balloon_dicts(dataset_path + "train")
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow('image', out.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    # Fine-tuning a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("data_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.MASK_ON = False

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()