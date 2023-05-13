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
import os
import yaml
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def getpreferredencoding(do_setlocale=True):
    return "UTF-8"


locale.getpreferredencoding = getpreferredencoding
parser = argparse.ArgumentParser(description='Train a detection model with SuperGradients')
parser.add_argument('--project', type=str, default='Dataset', help='name of the project')
parser.add_argument('--weights', type=str, default='Dataset', help='name of the project')
parser.add_argument('--location', type=str, default='Dataset', help='location of the dataset')
parser.add_argument('--model-arch', type=str, default='yolo_nas_s', help='model architecture')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
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
    'train_images_dir': train_images_dir,
    'train_labels_dir': train_labels_dir,
    'val_images_dir': val_images_dir,
    'val_labels_dir': val_labels_dir,
    'test_images_dir': test_images_dir,
    'test_labels_dir': test_labels_dir,
    'classes': [data['names'][i] for i in range(data['nc'])]
}

BATCH_SIZE = args.batch_size
MAX_EPOCHS = args.max_epochs
MODEL_ARCH = args.model_arch
trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)

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
