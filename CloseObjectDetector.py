import subprocess
from ai2thor.controller import Controller
from pynput import keyboard
from datetime import datetime
import time
import cv2
import os
from PIL import ImageTk, Image
import json
import numpy as np
from natsort import natsorted
import networkx as nx
import ast


import greedylookahead
import relationGraph
import setObjectPoses
import findDifferenceBetweenFrames_modified as fdbf


controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan1",

    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,

    renderDepthImage=True,
    renderInstanceSegmentation=True,

    width=720,#1920
    height=720,#950
    fieldOfView=90
)

object_dict ={
 0: 'Apple', 1: 'Apple_sliced', 2: 'Book', 3: 'Book_opened', 4: 'Bottle',
 5: 'Bottle_filled', 6: 'Bowl', 7: 'Bowl_filled', 8: 'Bread', 9: 'Bread_cooked_sliced',
 10: 'Bread_sliced', 11: 'ButterKnife', 12: 'Cabinet', 13: 'Cabinet_opened',
 14: 'CoffeeMachine', 15: 'CounterTop', 16: 'CreditCard', 17: 'Cup', 18: 'Cup_filled',
 19: 'DishSponge', 20: 'Drawer', 21: 'Drawer_opened', 22: 'Egg', 23: 'Egg_cooked_sliced',
 24: 'Egg_sliced', 25: 'Faucet', 26: 'Floor', 27: 'Fork', 28: 'Fridge', 29: 'Fridge_opened',
 30: 'GarbageCan', 31: 'HousePlant', 32: 'Kettle', 33: 'Knife', 34: 'Lettuce',
 35: 'Lettuce_sliced', 36: 'LightSwitch', 37: 'Microwave', 38: 'Microwave_opened',
 39: 'Mug', 40: 'Mug_filled', 41: 'Pan', 42: 'PaperTowelRoll', 43: 'PepperShaker',
 44: 'Plate', 45: 'Pot', 46: 'Pot_filled', 47: 'Potato', 48: 'Potato_cooked', 
 49: 'Potato_cooked_sliced', 50: 'Potato_sliced', 51: 'SaltShaker', 52: 'Shelf', 
 53: 'ShelvingUnit', 54: 'Sink', 55: 'SoapBottle', 56: 'Spatula', 57: 'Spoon', 
 58: 'Statue', 59: 'Stool', 60: 'StoveBurner', 61: 'StoveKnob', 62: 'Toaster',
 63: 'Tomato', 64: 'Tomato_sliced', 65: 'Vase', 66: 'Window', 67: 'WineBottle', 
 68: 'WineBottle_filled'}

reverse_action = {'MoveRight': 'MoveLeft',
                  'MoveLeft': 'MoveRight',
                  'MoveAhead': 'MoveBack',
                  'MoveBack': 'MoveAhead',
                  'LookDown': 'LookUp',
                  'LookUp': 'LookDown',
                  'RotateRight 10': 'RotateLeft 10',
                  'RotateLeft 10': 'RotateRight 10',
                  'Teleport': 'Teleport',
                  'Teleport2': 'Teleport2',
                  'Nothing' : 'Nothing'}


def find_key_by_object(obj, obj_dict):
    for key, value in obj_dict.items():
        if value == obj:
            return key
    return None

def get_next_value(d, current_key):
    keys = list(d.keys())
    try:
        current_index = keys.index(current_key)
        next_index = current_index + 1
        if next_index < len(keys):
            next_key = keys[next_index]
            return d[next_key]
        else:
            return None
    except ValueError:
        return None

def get_depth_value(xc, yc, depth_frame):
    xc = int(xc*720)
    yc = int(yc*720)
    return (depth_frame[xc, yc]/5)
       

def find_seq_tool(obj_num, obj_tools, task):
    obj_name = object_dict [obj_num]
    for successor in relation_graph.successors(obj_name):
        if relation_graph[obj_name][successor]['action'][0] == task:
            possible_tools = relation_graph[obj_name][successor]['action'][1]
    
    for i in range(len(possible_tools)):
        if possible_tools[i] == 'Hand':
            main_tool = 'Hand'
            break
        elif find_key_by_object(possible_tools[i], object_dict) in obj_tools:
            main_tool = possible_tools [i]
            break
    if main_tool:
        return main_tool
    else:
        return None
    
 
def adjust_seq_num(destination):
    global direction_flag
    global seq_num
    global sequence
    global frame_num
    temp_flagd = direction_flag
    direct = destination - frame_num  
    if direct < 0:
        direct += len(sequence)
    reverse = len(sequence) - direct
    if (direct <= reverse):
        direction_flag = True
    else:
        direction_flag = False     
    if temp_flagd != direction_flag:
        if temp_flagd:
            seq_num += 1
        else:
            seq_num -= 1    
                
                                         
def reach_frame(des_frame):
    global sequence
    global frame_num 
    global seq_num
    global direction_flag
    if frame_num == 0:
        seq_num = 1
        controller.step(
        action="Teleport",
        position=dict(x=1, y=0.9, z=-1.5),
        rotation=dict(x=0, y=180, z=0),
        horizon=0,
        standing=True) 
    
    if frame_num == des_frame:
        return
    
    direct = des_frame - frame_num
    if direct < 0:
        direct += len(sequence)
    reverse = len(sequence) - direct
    if (direct <= reverse):
        direction_flag = True
    else:
        direction_flag = False
    
    
    while (frame_num != des_frame):
    
        # For the case that we have to reverse from start
        if frame_num == 0 and direction_flag == False:
            frame_num = len(sequence) - 1
            seq_num = len(sequence)     
            
        if direction_flag:
            frame_num += 1 
            seq_num += 1
        else:
            frame_num -= 1 
            seq_num -= 1    
        # Choosing the action based on direction
        if direction_flag:
            action = sequence[seq_num] 
        else:
            action = reverse_action[sequence[seq_num]]      

        if action == "Teleport":
            controller.step(
            action="Teleport",
            position=dict(x=1, y=0.9, z=-1.5),
            rotation=dict(x=0, y=180, z=0),
            horizon=0,
            standing=True) 
                       
        elif action == 'Teleport2':
            controller.step(
            action="Teleport",
            position=dict(x=1.25, y=0.900999903678894, z=-1.75),
            rotation=dict(x = -0.0, y = 180.0000457763672, z = 0.0),
            horizon=1.5309450418499182e-06,
            standing=True) 
                 
        elif action == "RotateLeft 10":
            controller.step(
                action="RotateLeft",
                degrees= 10
            )

        elif action == "RotateRight 10":
            controller.step(
                action="RotateRight",
                degrees= 10
            )

        elif action == "Nothing":
            pass
        else:
            controller.step(action)   
        
        capture_frame()
   
      
    print("Reached Destination Frame.")
    
def capture_frame():
    event = controller.last_event
    frame = event.frame
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_path = os.path.join('frames', f'{frame_num}.jpg')
    cv2.imwrite(os.path.join('frames', f'{curr_time}_{frame_num}.jpg'), frame_rgb)
    
    
def slice_objects(actions_obj):   
    for key, value in actions_obj.items():
        if 'slice' in value:
            reach_frame(best_frames1[key][0])
            controller.step(action='SetObjectStates', SetObjectStates={'objectType': object_dict[key], 'stateChange': 'sliceable', 'isSliced': True})

    
    return
            
def check_frame(des_frame_info):
    tool_flag = False
    pick_flag = False
    tool_name = ''
    global frame_num
    global seq_num
    global direction_flag
    global sequence
    
    reach_frame(des_frame_info[0])
        
    if (frame_num == des_frame_info[0]):
        
        # Get the object_id from object_name using metadata of our current frame

        query = controller.step(
                action="GetObjectInFrame",
                x= des_frame_info[1] ,
                y= des_frame_info[2],
                checkVisible=False
            )
        
        if query.metadata["actionReturn"] != None:
                object_id = query.metadata["actionReturn"]
                        

        event = controller.step(
                action="PickupObject",
                objectId= object_id,
                forceAction=False,
                manualInteract=False
            )   
        
        controller.step("LookDown")
        controller.step("LookDown")
        # adjust_seq_num(des_frame_info[0] + 1)
        capture_frame()     

        controller.step("MoveHeldObjectDown",moveMagnitude=0.5)         
        mov_val = abs(0.5 - des_frame_info[1])       
        capture_frame()
               
        controller.step(
                action="RotateHeldObject",
                pitch=0,
                yaw=90,
                roll=0
        )    
        capture_frame()

        controller.step("MoveHeldObjectAhead",moveMagnitude=0.5)        
        capture_frame()
        pick_flag = True 
        capture_frame()
        

        controller.step("MoveHeldObjectUp",moveMagnitude=1)           
        capture_frame()
            

        controller.step("MoveHeldObjectAhead",moveMagnitude=0.5)        
        capture_frame()

        controller.step(
                action="RotateHeldObject",
                pitch=0,
                yaw=90,
                roll=0
        )    
        capture_frame()        
        controller.step(
        action="DropHandObject",
        forceAction=True
        )   
        pick_flag = False
        # depth_data = event.depth_frame
        # np.savetxt(f'frames/depth_{frame_num}.txt', depth_data, fmt='%.6f')
        # adjust_seq_num(des_frame_info[0] + 1)
        controller.step("LookUp")
        controller.step("LookUp")
    print('')

 
def read_missing_objects(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    scene_data = {}
    current_scene = None

    for line in lines:
        line = line.strip()
        if line.startswith("Missing Objects Data for Scene"):
            # Extract scene number and remove any trailing characters like ':'
            current_scene = int(line.split()[-1].rstrip(':'))
            scene_data[current_scene] = []
        elif current_scene is not None:
            try:
                # Convert the line to a list and extend the current scene's data
                data = ast.literal_eval(line)
                scene_data[current_scene].extend(data)  # Use extend instead of append
            except (SyntaxError, ValueError) as e:
                print(f"Error parsing line: {line}. Error: {e}")

    # Convert the dictionary values to 2D arrays
    scene1_data = scene_data.get(1, [])
    scene2_data = scene_data.get(2, [])

    return scene1_data, scene2_data



# SOME IMPORTANT GLOBAL VARIABLES
relation_graph = relationGraph.create_relation_graph()
direction_flag = False
seq_num = 0   
frame_num = 0
best_frames1 = {i: [0, 0, 0, 0, 0] for i in range(69)}
best_frames2 = {i: [0, 0, 0, 0, 0] for i in range(69)}
sequence = []
knife_moved = False
with open('seq_action.txt', 'r') as file:
    sequence = file.read().splitlines()  
seq_length = len(sequence)
    
    
if __name__ == "__main__":

    setObjectPoses.set_object_poses_from_metadata(controller, "metadata1.json")
    scene1_missing, scene2_missing = read_missing_objects('sample.txt')

    # Print the results
    print("Scene 1 Data:", scene1_missing)
    print("Scene 2 Data:", scene2_missing)

    for frame_info in scene1_missing:
        check_frame(frame_info)   
        
                