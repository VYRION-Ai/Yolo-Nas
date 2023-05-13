Welcome to the YOLO-NAS Repository!
![image](https://github.com/totoadel/Yolo-Nas/assets/23275255/6894bb61-3e84-46ef-af81-a5f69a9ee306)

Object Detection

![image](https://github.com/totoadel/Yolo-Nas/assets/23275255/8a317ed3-2c3c-4fdb-ba44-be7e3e25db67)

Our architecture is designed to deliver unparalleled accuracy-speed performance, pushing the boundaries of what's possible in object detection. In this repository, we provide instructions for training your own model using our cutting-edge architecture.

## Getting Started

To begin your training journey, first install the `super-gradients` library version 3.1.1 with the following command:

```
!pip install -q super-gradients==3.1.1
```

After installation is complete, don't forget to restart the runtime by navigating to `Runtime -> Restart runtime` and confirming with a simple click of "Yes".

Next, clone this repository with the following command:

```
!git clone  https://github.com/VYRION-Ai/Yolo-Nas.git
```

You'll then want to add your dataset to the mix. To do this, use the following code snippet with Google Colab:

```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/
%cp /content/drive/MyDrive/yolo_dataset.zip /content/
!unzip  /content/yolo_dataset.zip
!rm  /content/yolo_dataset.zip
```

And finally, it's time to start training! Navigate to the YOLO-NAS directory and run the following command to begin your project:

```
%cd /content/Yolo-Nas
!python train.py --project "Dataset"  --data /content/Dataset/data.yaml --location '/content/Dataset' --model-arch yolo_nas_s --batch-size 16 --max-epochs 25 --checkpoint-dir /content/checkpoints
```
Validation
```
%cd /content/Yolo-Nas
!python valid.py  --data /content/Dataset/data.yaml --location '/content/Dataset' --weights /content/checkpoints/Dataset/ckpt_best.pth
```
## Contributing

We welcome contributions from the community! If you encounter any issues or have any suggestions for improving the YOLO-NAS architecture, please feel free to open an issue or submit a pull request.

## License

This repository is licensed under the MIT license. See `LICENSE` for more information.
