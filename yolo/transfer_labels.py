#%%
import json
import os
from tkinter import filedialog

from tomni.transformers import contours2json


label_file = filedialog.askopenfile(title="load label_file")
location_new_labels = filedialog.askdirectory(title="New location labels")
local_data = os.path.dirname(label_file.name)
labels = json.loads(label_file.read())

all_catogories = [
    "red blood cell",
    "gametocyte",
    "leukocyte",
    "ring",
    "schizont",
    "trophozoite",
    "difficult",
]
for label in labels:
    image_name = label["image"]["pathname"].replace("/images/", "").replace(".png", "")

    json_objects = {cat: [] for cat in all_catogories}
    for org_object in label["objects"]:
        x_min = org_object["bounding_box"]["minimum"]["c"]
        y_min = org_object["bounding_box"]["minimum"]["r"]
        x_max = org_object["bounding_box"]["maximum"]["c"]
        y_max = org_object["bounding_box"]["maximum"]["r"]

        contour = [
            [[[x_min, y_min]], [[x_max, y_min]], [[x_max, y_max]], [[x_min, y_max]]]
        ]

        category = org_object["category"]

        json_object = contours2json(contour)[0]
        json_objects[category].append(json_object)

    with open(
        os.path.join(location_new_labels.replace("/", "\\"), image_name + ".json"), "w"
    ) as f:
        json.dump(json_objects, f)
