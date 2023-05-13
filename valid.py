import argparse
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
# python train.py --project "pane" --data /content/data.yaml --location '/content/Dataset' --model-arch yolo_nas_s --batch-size 16 --max-epochs 25 --checkpoint-dir /content/checkpoints
import locale
from  google.colab.patches import cv2_imshow
import os
import yaml
import sys
from pathlib import Path
import supervision as sv
import os
import torch
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from onemetric.cv.object_detection import ConfusionMatrix
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding
parser = argparse.ArgumentParser(description='Train a detection model with SuperGradients')
parser.add_argument('--weights', type=str, default='average_model.pth', help='name of the project')
parser.add_argument('--location', type=str, default='Dataset', help='location of the dataset')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--model-arch', type=str, default='yolo_nas_s', help='model architecture')
parser.add_argument('--project', type=str, default='Dataset', help='name of the project')
parser.add_argument('--checkpoint-dir', type=str, default=ROOT / 'checkpoints', help='directory to save checkpoints')
parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
args = parser.parse_args()
experiment_name = args.project
ckpt_root_dir = args.checkpoint_dir
# Initialize default directory paths
test_images_dir = None
test_labels_dir = None

with open(args.data) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    test_paths = data.get('test', [])
    test_images_dir = test_paths
    test_labels_dir = test_paths.replace("images", "labels")

dataset_params = {
    'data_dir': args.location,
    'test_images_dir': test_images_dir,
    'test_labels_dir': test_labels_dir,
    'classes': [data['names'][i] for i in range(data['nc'])]
}

MODEL_ARCH = args.model_arch
BATCH_SIZE = args.batch_size

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)
best_model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params['classes']),
    checkpoint_path=args.weights
).to(DEVICE)
trainer.test(
    model=best_model,
    test_loader=test_data,
    test_metrics_list=DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        num_cls=len(dataset_params['classes']),
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=0.7
        )
    )
)

ds = sv.Dataset.from_yolo(
    images_directory_path=dataset_params['test_images_dir'],
    annotations_directory_path=dataset_params['test_labels_dir'],
    data_yaml_path=args.data,
    force_masks=False
)
CONFIDENCE_TRESHOLD = 0.5

predictions = {}

for image_name, image in ds.images.items():
    result = list(best_model.predict(image, conf=CONFIDENCE_TRESHOLD))[0]
    detections = sv.Detections(
        xyxy=result.prediction.bboxes_xyxy,
        confidence=result.prediction.confidence,
        class_id=result.prediction.labels.astype(int)
    )
    predictions[image_name] = detections
random.seed(10)

#####################################33
MAX_IMAGE_COUNT = 5

n = min(MAX_IMAGE_COUNT, len(ds.images))

keys = list(ds.images.keys())
keys = random.sample(keys, n)

box_annotator = sv.BoxAnnotator()

images = []
titles = []

for key in keys:
    frame_with_annotations = box_annotator.annotate(
        scene=ds.images[key].copy(),
        detections=ds.annotations[key],
        skip_label=True
    )
    images.append(frame_with_annotations)
    titles.append('annotations')
    frame_with_predictions = box_annotator.annotate(
        scene=ds.images[key].copy(),
        detections=predictions[key],
        skip_label=True
    )
    images.append(frame_with_predictions)
    titles.append('predictions')
for i, (image, title) in enumerate(zip(images, titles)):
    filename = f"{title}.jpg"  # use title as the filename
    cv2.imwrite(filename, image)  # save the image to disk

keys = list(ds.images.keys())

annotation_batches, prediction_batches = [], []
for key in keys:
    annotation=ds.annotations[key]
    annotation_batch = np.column_stack((
        annotation.xyxy,
        annotation.class_id
    ))
    annotation_batches.append(annotation_batch)
    prediction=predictions[key]
    prediction_batch = np.column_stack((
        prediction.xyxy,
        prediction.class_id,
        prediction.confidence
    ))
    prediction_batches.append(prediction_batch)
#
confusion_matrix = ConfusionMatrix.from_detections(
    true_batches=annotation_batches,
    detection_batches=prediction_batches,
    num_classes=len(ds.classes),
    conf_threshold=CONFIDENCE_TRESHOLD
)
#
confusion_matrix.plot(os.path.join(os.getcwd(), "confusion_matrix.png"), class_names=ds.classes)
