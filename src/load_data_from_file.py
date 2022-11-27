import os
from scipy.io.wavfile import read
import json


def load_data_from_file(
    data_dir: str = "data/audio_mnist", labels_to_load: list = [0, 1]
) -> None:
    data_dict = {"signals": [], "frequency": [], "label": []}
    for folder in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, folder)):
            continue
        else:
            for file in os.listdir(f"{data_dir}/{folder}"):
                if int(file.split("_")[0]) in labels_to_load:
                    frequency, signal = read(f"{data_dir}/{folder}/{file}")
                    data_dict["signals"].append(signal.tolist())
                    data_dict["frequency"].append(frequency)
                    data_dict["label"].append(int(file.split("_")[0]))
                    data_dict["speaker"].append(int(file.split("_")[1]))

    with open(os.path.join(data_dir, f"loaded_data_{labels_to_load}.json"), "w") as fp:
        json.dump(data_dict, fp)
