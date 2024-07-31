import prior
import os
import random
import json
from collections import Counter
from PIL import Image
from ai2thor.controller import Controller
from thortils import VideoRecorder 

from rearrange.procthor_rearrange.constants import ASSET_TO_OBJECT , PICKUPABLE_OBJECTS_PROC
from cospomdp_apps.thor.rearrange_utils import get_obj_from_asset_ID

import copy

def get_top_down_frame(controller):
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




def try_video_recording(controller):

    fps = 5
    folder_path = "."

    os.makedirs(folder_path, exist_ok=True)
    filename = "firstperson.mp4"
    recorder = VideoRecorder(
                vid_path= os.path.join(folder_path, filename),
                fps=fps)
    filename = "top_view.mp4"
    top_recorder = VideoRecorder(
                vid_path= os.path.join(folder_path, filename),
                fps=fps)

    frame = Image.fromarray(controller.last_event.frame)

    event = controller.step(action="RotateRight")

    top_frame = get_top_down_frame(controller)
    
    for i in range(55):

        action = random.choice(["MoveAhead", "RotateRight", "RotateLeft"])
        event = controller.step(action=action)

        frame = Image.fromarray(controller.last_event.frame)
        top_frame_2 = Image.fromarray(event.third_party_camera_frames[-1])

        recorder.add(frame)
        #top_recorder.add(top_frame)
        top_recorder.add(top_frame_2)

    recorder.finish()
    top_recorder.finish()

def inspect_houses(dataset, controller):
    for house in dataset['train']:
        controller.reset(scene=house)
        top_down_frame = get_top_down_frame(controller)
        #print (house)
        event = controller.step(action="RotateRight")
        event = controller.step(action="RotateRight")
        event = controller.step(action="RotateRight")
        print ("num rooms: ", len(house['rooms']))


def test_scenes():
    #dataset = prior.load_dataset("procthor-10k", revision='rearrangement-2022')
    dataset = prior.load_dataset("procthor-10k")#, revision='rearrangement-2022')
    house = dataset["train"][9]
    controller = Controller(scene=house)
    top_down_frame = get_top_down_frame(controller)

    house = dataset["train"][10]
    controller = Controller(scene=house)
    top_down_frame = get_top_down_frame(controller)

    house = dataset["train"][11]
    controller = Controller(scene=house)
    #try_video_recording(controller)
    top_down_frame = get_top_down_frame(controller)
    
    inspect_houses(dataset, controller)


def process_house_data(houses, split_name):
    result = {}
    
    for house_index, house in enumerate(houses):

        house_number = f"{split_name}_{house_index}"
        object_types = [get_obj_from_asset_ID(obj['assetId'],True) for obj in house['objects']]
        house_info = {
            "house_number": f"{split_name}_{house_index}",
            "num_rooms": len(house['rooms']),
            "num_moveable_objects": sum(1 for obj_type in object_types if obj_type not in PICKUPABLE_OBJECTS_PROC ),
            "has_duplicate_objects": False
        }
        
        # Count object types
        object_counts = Counter(object_types)
        
        # Check if any object type appears more than once
        if any(count > 1 for count in object_counts.values()):
            house_info["has_duplicate_objects"] = True
        
        result[house_number] = house_info
    
    return result

def write_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON file '{filename}' has been created successfully.")

def save_houses_info(revision):

    if revision is None:
        dataset = prior.load_dataset("procthor-10k")#, revision='rearrangement-2022')
    else :
        dataset = prior.load_dataset("procthor-10k", revision=revision)


    for split in ['train', 'val', 'test']:
        processed_data = process_house_data(dataset[split], split)
        write_to_json(processed_data, f'house_data_summary_{split}.json')

def load_json_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def filter_houses(data, min_rooms=1, max_rooms=4, allow_duplicates=False):
    filtered_houses = []
    for house_id, house_info in data.items():
        if (min_rooms <= house_info['num_rooms'] <= max_rooms and
            (allow_duplicates or not house_info['has_duplicate_objects'])):
            filtered_houses.append(house_id)
    return filtered_houses

def filter_and_save_houses_by_split(min_rooms=1, max_rooms=4, allow_duplicates=False):
    splits = ['train', 'val', 'test']
    data = {split: load_json_file(f'house_data_summary_{split}.json') for split in splits}

    # Filter houses for each split
    filtered_results = {}
    for split in splits:
        filtered_houses = filter_houses(data[split], min_rooms, max_rooms, allow_duplicates)
        filtered_results[split] = {
            "total_houses": len(filtered_houses),
            "house_ids": filtered_houses
        }

    # Calculate overall total
    overall_total = sum(result["total_houses"] for result in filtered_results.values())

    # Prepare final results
    final_results = {
        "overall_total": overall_total,
        "splits": filtered_results
    }

    # Print results
    print(f"Overall total number of houses meeting criteria: {overall_total}")
    for split in splits:
        print(f"\n{split.capitalize()} split:")
        print(f"Total houses: {filtered_results[split]['total_houses']}")
        print("House IDs:")
        for house_id in filtered_results[split]['house_ids']:
            print(house_id)

    # Save the results to a new JSON file
    with open(f'filtered_houses_by_split_{min_rooms}_{allow_duplicates}.json','w') as f:
        json.dump(final_results, f, indent=2)
    print("\nResults saved to 'filtered_houses_by_split.json'")

if __name__ == '__main__':
    controller = Controller()
    save_houses_info(revision='rearrangement-2022')
    #allow_duplicates = True
    #for i in range(1, 5):
    #    filter_and_save_houses_by_split(min_rooms=i, max_rooms=i, allow_duplicates=allow_duplicates)
    #test_scenes()