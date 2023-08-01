import os, json
import copy
from ai2thor.controller import Controller
from PIL import Image
import numpy as np
import prior

# import pdb; pdb.set_trace()

# load house specifiction
dataset = prior.load_dataset("procthor-10k")
house_spec = dataset['train'][0]

# # load house specifiction
# house_json_path = 'big-dataset/train/535858-5341.json'
# house_spec = json.load(open(house_json_path, "r"))

print(type(house_spec), house_spec.keys())

# initialize controller
controller = Controller(scene=house_spec, gpu_device=0, quality="Low")

# render egocentric view
img = Image.fromarray(controller.last_event.frame)
img.save('debug/egocentric.png')

# navigation action
event = controller.step(action="RotateRight")
img = Image.fromarray(event.frame)
img.save('debug/rotate_right.png')

# get top down view
def get_top_down_frame():
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


img = get_top_down_frame()
img.save('debug/top_down.png')