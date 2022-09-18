import fiftyone as fo

fo.config.desktop_app = True

dataset_name = "tweet_dataset"
dataset = fo.load_dataset(dataset_name)
session = fo.launch_app(dataset, port=5151)