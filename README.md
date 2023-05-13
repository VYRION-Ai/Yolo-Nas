# Yolo-Nas
Ready to deploy pre-trained SOTA models

YOLO-NAS architecture is out! The new YOLO-NAS delivers state-of-the-art performance with the unparalleled accuracy-speed performance, outperforming other models such as YOLOv5, YOLOv6, YOLOv7 and YOLOv8. Check it out here: YOLO-NAS.

# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

All Computer Vision Models - Pretrained Checkpoints can be found in the Model Zoo
Classification
Semantic Segmentation
Object Detection
Easy to train SOTA Models

Easily load and fine-tune production-ready, pre-trained SOTA models that incorporate best practices and validated hyper-parameters for achieving best-in-class accuracy. For more information on how to do it go to Getting Started
Plug and play recipes

python -m super_gradients.train_from_recipe architecture=regnetY800 dataset_interface.data_dir=<YOUR_Imagenet_LOCAL_PATH> ckpt_root_dir=<CHEKPOINT_DIRECTORY>

More example on how and why to use recipes can be found in Recipes
Production readiness

All SuperGradients models’ are production ready in the sense that they are compatible with deployment tools such as TensorRT (Nvidia) and OpenVINO (Intel) and can be easily taken into production. With a few lines of code you can easily integrate the models into your codebase.

# Load model with pretrained weights
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

# Prepare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 640, 640])
    
# Create dummy_input

# Convert model to onnx
torch.onnx.export(model, dummy_input,  "yolo_nas_m.onnx")

More information on how to take your model to production can be found in Getting Started notebooks
Quick Installation

pip install super-gradients

What's New - Version 3.1.1 (May 3rd)

    YOLO-NAS
    New predict function (predict on any image, video, url, path, stream)
    RoboFlow100 datasets integration
    A new Documentation Hub
    Integration with DagsHub for experiment monitoring
    Support Darknet/Yolo format detection dataset (used by Yolo v5, v6, v7, v8)
    Segformer model and recipe
    Post Training Quantization and Quantization Aware Training - notebooks

Check out SG full release notes.
