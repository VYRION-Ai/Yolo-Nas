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
parser.add_argument('--project', type=str, default='pane', help='name of the project')
parser.add_argument('--location', type=str, default='Dataset', help='location of the dataset')
parser.add_argument('--model-arch', type=str, default='yolo_nas_s', help='model architecture')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--max-epochs', type=int, default=25, help='maximum number of epochs')
parser.add_argument('--checkpoint-dir', type=str, default=ROOT / 'checkpoints', help='directory to save checkpoints')
parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
args = parser.parse_args()

experiment_name = args.project
ckpt_root_dir = args.checkpoint_dir

with open(args.project) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

train_paths = data['train']
for train_path in train_paths:
    if 'images/train' in train_path:
        train_images_dir = os.path.join(data['path'], train_path)
        train_labels_dir = os.path.join(data['path'], 'labels/train')
        break
    elif 'train/images' in train_path:
        train_images_dir = os.path.join(data['path'], train_path)
        train_labels_dir = os.path.join(data['path'], 'train/labels')
        break

val_paths = data['val']
for val_path in val_paths:
    if 'images/valid' in val_path:
        val_images_dir = os.path.join(data['path'], val_path)
        val_labels_dir = os.path.join(data['path'], 'labels/valid')
        break
    elif 'valid/images' in val_path:
        val_images_dir = os.path.join(data['path'], val_path)
        val_labels_dir = os.path.join(data['path'], 'valid/labels')
        break

test_paths = data.get('test', [])
if len(test_paths) > 0:
    for test_path in test_paths:
        if 'images/test' in test_path:
            test_images_dir = os.path.join(data['path'], test_path)
            test_labels_dir = os.path.join(data['path'], 'labels/test')
            break
        elif 'test/images' in test_path:
            test_images_dir = os.path.join(data['path'], test_path)
            test_labels_dir = os.path.join(data['path'], 'test/labels')
            break
else:
    test_images_dir = None
    test_labels_dir = None

dataset_params = {
    'data_dir': data['path'],
    'train_images_dir': train_images_dir,
    'train_labels_dir': train_labels_dir,
    'val_images_dir': val_images_dir,
    'val_labels_dir': val_labels_dir,
    'test_images_dir': test_images_dir,
    'test_labels_dir': test_labels_dir,
    'classes': [data['names'][i] for i in range(data['nc'])]
}

# dataset_params = {
#     'data_dir': args.location,
#     'train_images_dir': 'images/train',
#     'train_labels_dir': 'labels/train',
#     'val_images_dir': 'images/valid',
#     'val_labels_dir': 'labels/valid',
#     'test_images_dir': 'images/test',
#     'test_labels_dir': 'labels/test',
#     'classes': [args.project]
# }
BATCH_SIZE = args.batch_size
MAX_EPOCHS = args.max_epochs
MODEL_ARCH = args.model_arch
trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=ckpt_root_dir)

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size': BATCH_SIZE,
        'num_workers': 2
    }
)

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

model = models.get(
    MODEL_ARCH,
    num_classes=len(dataset_params['classes']),
    pretrained_weights="coco"
)

train_params = {
    'silent_mode': False,
    "average_best_models": True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    "max_epochs": MAX_EPOCHS,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
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
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)
