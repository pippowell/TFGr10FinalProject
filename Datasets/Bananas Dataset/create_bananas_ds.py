import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["banana"],
    max_samples=100
)

session = fo.launch_app(dataset)

session.dataset = dataset

