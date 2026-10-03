from copy import deepcopy
import math
import re

COLORS = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "orange": "orange",
    "purple": "purple",
    "pink": "pink",
    "brown": "brown",
    "gold": "gold",
    "tan": "tan",
    "cyan": "cyan",
    "silver": "silver",
    "lightblue": "lightblue",
    "darkgreen": "darkgreen",
}

ALIASES = {
    "cube": ("cube",),
    "box": ("box", "block", "rectangular box"),
    "sphere": ("sphere", "ball"),
    "cylinder": ("cylinder", "can"),
    "cone": ("cone",),
    "pyramid": ("pyramid",),
    "pencil": ("pencil",),
    "eraser": ("eraser",),
    "ruler": ("ruler",),
    "table": ("table", "desk"),
    "chair": ("chair", "seat"),
    "house": ("house", "home"),
    "car": ("car", "automobile", "sedan"),
    "truck": ("truck", "pickup"),
    "bus": ("bus",),
    "plane": ("plane", "airplane", "aircraft", "jet"),
    "boat": ("boat", "ship"),
    "helicopter": ("helicopter", "chopper"),
    "rocket": ("rocket", "spaceship", "space ship"),
    "train": ("train", "locomotive"),
    "motorcycle": ("motorcycle", "motorbike"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "bookshelf": ("bookshelf", "bookcase"),
    "lamp": ("lamp", "light"),
    "tree": ("tree",),
    "flower": ("flower",),
    "person": ("person", "human", "man", "woman"),
    "bird": ("bird",),
    "cat": ("cat",),
    "dog": ("dog",),
}

ALIASES.update({
    "bicycle": ("bicycle", "bike"), "traffic_light": ("traffic light",),
    "fire_hydrant": ("fire hydrant",), "stop_sign": ("stop sign",),
    "parking_meter": ("parking meter",), "bench": ("bench",),
    "horse": ("horse",), "sheep": ("sheep",), "cow": ("cow",),
    "elephant": ("elephant",), "bear": ("bear",), "zebra": ("zebra",),
    "giraffe": ("giraffe",), "backpack": ("backpack",),
    "umbrella": ("umbrella",), "handbag": ("handbag",), "tie": ("tie",),
    "suitcase": ("suitcase",), "frisbee": ("frisbee",), "skis": ("skis",),
    "snowboard": ("snowboard",), "sports_ball": ("sports ball",),
    "kite": ("kite",), "baseball_bat": ("baseball bat",),
    "baseball_glove": ("baseball glove",), "skateboard": ("skateboard",),
    "surfboard": ("surfboard",), "tennis_racket": ("tennis racket",),
    "bottle": ("bottle",), "wine_glass": ("wine glass",), "cup": ("cup",),
    "fork": ("fork",), "knife": ("knife",), "spoon": ("spoon",),
    "bowl": ("bowl",), "banana": ("banana",), "apple": ("apple",),
    "orange": ("orange",), "sandwich": ("sandwich",),
    "broccoli": ("broccoli",), "carrot": ("carrot",),
    "hot_dog": ("hot dog",), "pizza": ("pizza",),
    "donut": ("donut",), "cake": ("cake",), "potted_plant": ("potted plant",),
    "dining_table": ("dining table",), "toilet": ("toilet",),
    "tv": ("tv", "television"),
    "laptop": ("laptop",), "mouse": ("mouse",), "remote": ("remote",),
    "keyboard": ("keyboard",), "cell_phone": ("cell phone", "cellphone", "phone"),
    "microwave": ("microwave",), "oven": ("oven",), "toaster": ("toaster",),
    "sink": ("sink",), "refrigerator": ("refrigerator", "fridge"),
    "book": ("book",), "clock": ("clock",), "vase": ("vase",),
    "scissors": ("scissors",), "teddy_bear": ("teddy bear",),
    "hair_drier": ("hair drier", "hair dryer"), "toothbrush": ("toothbrush",),
})

# Small helpers
def _part(part_id, kind, dimensions, color, parent=None, **offsets):
    item = {
        "id": part_id,
        "type": kind,
        "dimensions": dimensions,
        "color": color,
        "attach_to": parent,
        "position": "center",
    }
    item.update(offsets)
    return item

def _wheel(part_id, radius, width, parent, x, y, z, color="black"):
    # Wheel with flat face along Y, for any vehicle that rolls along X.
    return _part(
        part_id,
        "cylinder",
        {"radius": radius, "height": width},
        color,
        parent,
        position="center",
        rotation={"rx": 90},
        offset_x=x,
        offset_y=y,
        offset_z=z,
    )

# Vehicles
def _car_model(color):
    body_dz = 0.65
    wheel_r = 0.42
    wheel_w = 0.28
    wheel_z = -body_dz / 2 + 0.10
    wheel_x = 1.45
    wheel_y = 0.92
    return [
        _part("body", "cube", {"dx": 4.6, "dy": 1.9, "dz": body_dz}, color),
        _part("bumper_front", "cube", {"dx": 0.12, "dy": 1.88, "dz": 0.22},
              "darkgray", "body", position="right"),
        _part("bumper_rear", "cube", {"dx": 0.12, "dy": 1.88, "dz": 0.22},
              "darkgray", "body", position="left"),
        _part("cabin", "cube", {"dx": 2.30, "dy": 1.70, "dz": 0.70},
              color, "body", position="top", offset_x=-0.15),
        _part("windshield", "cube", {"dx": 0.06, "dy": 1.58, "dz": 0.62},
              "lightblue", "cabin", position="right"),
        _part("rear_window", "cube", {"dx": 0.06, "dy": 1.58, "dz": 0.58},
              "lightblue", "cabin", position="left"),
        _part("window_left", "cube", {"dx": 1.85, "dy": 0.04, "dz": 0.48},
              "lightblue", "cabin", position="front", offset_z=0.05),
        _part("window_right", "cube", {"dx": 1.85, "dy": 0.04, "dz": 0.48},
              "lightblue", "cabin", position="back", offset_z=0.05),
        _part("mirror_left", "cube", {"dx": 0.14, "dy": 0.06, "dz": 0.11},
              "black", "cabin", position="front",
              offset_x=0.95, offset_z=-0.02),
        _part("mirror_right", "cube", {"dx": 0.14, "dy": 0.06, "dz": 0.11},
              "black", "cabin", position="back",
              offset_x=0.95, offset_z=-0.02),
        _part("headlight_left", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.16},
              "yellow", "body", position="right",
              offset_y=0.68, offset_z=0.06),
        _part("headlight_right", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.16},
              "yellow", "body", position="right",
              offset_y=-0.68, offset_z=0.06),
        _part("taillight_left", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.14},
              "red", "body", position="left",
              offset_y=0.68, offset_z=0.06),
        _part("taillight_right", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.14},
              "red", "body", position="left",
              offset_y=-0.68, offset_z=0.06),
        _wheel("wheel_fl", wheel_r, wheel_w, "body", wheel_x, wheel_y, wheel_z),
        _wheel("wheel_fr", wheel_r, wheel_w, "body", wheel_x, -wheel_y, wheel_z),
        _wheel("wheel_rl", wheel_r, wheel_w, "body", -wheel_x, wheel_y, wheel_z),
        _wheel("wheel_rr", wheel_r, wheel_w, "body", -wheel_x, -wheel_y, wheel_z),
    ]

def _truck_model(color):
    chassis_dz = 0.35
    wheel_r = 0.52
    wheel_w = 0.34
    wheel_z = -chassis_dz / 2 - 0.05
    return [
        _part("chassis", "cube", {"dx": 5.6, "dy": 2.1, "dz": chassis_dz}, color),
        _part("cab", "cube", {"dx": 1.9, "dy": 2.05, "dz": 1.5}, color,
              "chassis", position="top", offset_x=1.75),
        _part("cab_windshield", "cube",
              {"dx": 0.06, "dy": 1.85, "dz": 0.72}, "lightblue",
              "cab", position="right"),
        _part("cab_side_window_left", "cube",
              {"dx": 0.85, "dy": 0.04, "dz": 0.65}, "lightblue",
              "cab", position="front", offset_x=0.35),
        _part("cab_side_window_right", "cube",
              {"dx": 0.85, "dy": 0.04, "dz": 0.65}, "lightblue",
              "cab", position="back", offset_x=0.35),
        _part("bed", "cube", {"dx": 3.2, "dy": 2.0, "dz": 1.1}, color,
              "chassis", position="top", offset_x=-1.0),
        _part("bed_gate", "cube", {"dx": 0.10, "dy": 2.0, "dz": 1.1},
              "darkgray", "bed", position="left"),
        _part("headlight_left", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.18},
              "yellow", "cab", position="right", offset_y=0.72, offset_z=-0.15),
        _part("headlight_right", "cube", {"dx": 0.08, "dy": 0.34, "dz": 0.18},
              "yellow", "cab", position="right", offset_y=-0.72, offset_z=-0.15),
        _wheel("wheel_fl", wheel_r, wheel_w, "chassis", 1.75, 1.05, wheel_z),
        _wheel("wheel_fr", wheel_r, wheel_w, "chassis", 1.75, -1.05, wheel_z),
        _wheel("wheel_rl1", wheel_r, wheel_w, "chassis", -1.0, 1.05, wheel_z),
        _wheel("wheel_rr1", wheel_r, wheel_w, "chassis", -1.0, -1.05, wheel_z),
        _wheel("wheel_rl2", wheel_r, wheel_w, "chassis", -1.9, 1.05, wheel_z),
        _wheel("wheel_rr2", wheel_r, wheel_w, "chassis", -1.9, -1.05, wheel_z),
    ]

def _bus_model(color):
    body_dz = 2.6
    wheel_r = 0.55
    wheel_w = 0.36
    wheel_z = -body_dz / 2 + 0.30
    return [
        _part("body", "cube", {"dx": 6.6, "dy": 2.35, "dz": body_dz}, color),
        _part("window_band", "cube",
              {"dx": 5.9, "dy": 2.37, "dz": 0.85}, "lightblue",
              "body", position="center", offset_z=0.55),
        _part("windshield", "cube", {"dx": 0.06, "dy": 2.1, "dz": 1.3},
              "black", "body", position="right"),
        _part("rear_window", "cube", {"dx": 0.06, "dy": 2.1, "dz": 1.0},
              "black", "body", position="left", offset_z=0.35),
        _part("door_front", "cube", {"dx": 0.9, "dy": 0.04, "dz": 1.9},
              "darkgray", "body", position="front", offset_x=1.6),
        _part("door_rear", "cube", {"dx": 0.9, "dy": 0.04, "dz": 1.9},
              "darkgray", "body", position="front", offset_x=-1.6),
        _wheel("wheel_fl", wheel_r, wheel_w, "body", 2.1, 1.1, wheel_z),
        _wheel("wheel_fr", wheel_r, wheel_w, "body", 2.1, -1.1, wheel_z),
        _wheel("wheel_rl", wheel_r, wheel_w, "body", -2.1, 1.1, wheel_z),
        _wheel("wheel_rr", wheel_r, wheel_w, "body", -2.1, -1.1, wheel_z),
    ]

def _motorcycle_model(color):
    wheel_r = 0.60
    return [
        _part("frame", "cube", {"dx": 2.1, "dy": 0.22, "dz": 0.30}, color),
        _part("fuel_tank", "cube", {"dx": 0.75, "dy": 0.45, "dz": 0.35},
              color, "frame", position="top", offset_x=0.35),
        _part("seat", "cube", {"dx": 0.95, "dy": 0.42, "dz": 0.16},
              "black", "frame", position="top", offset_x=-0.55),
        _part("handlebar", "cube", {"dx": 0.10, "dy": 0.78, "dz": 0.10},
              "black", "frame", position="top", offset_x=0.95, offset_z=0.60),
        _part("headlight", "sphere", {"radius": 0.14}, "yellow",
              "frame", position="right", offset_z=0.45),
        _wheel("wheel_front", wheel_r, 0.14, "frame", 1.05, 0, -0.55),
        _wheel("wheel_rear", wheel_r, 0.16, "frame", -1.05, 0, -0.55),
    ]

def _bicycle_model(color):
    wheel_r = 0.68
    return [
        _part("frame", "cube", {"dx": 1.9, "dy": 0.10, "dz": 0.10}, color),
        _part("seat", "cube", {"dx": 0.32, "dy": 0.16, "dz": 0.08},
              "black", "frame", position="top", offset_x=-0.6, offset_z=0.35),
        _part("handlebar", "cube", {"dx": 0.08, "dy": 0.55, "dz": 0.06},
              "black", "frame", position="top", offset_x=0.80, offset_z=0.45),
        _part("pedals", "cylinder", {"radius": 0.15, "height": 0.4},
              "black", "frame", rotation={"rx": 90}),
        _wheel("wheel_front", wheel_r, 0.06, "frame", 1.00, 0, -0.68),
        _wheel("wheel_rear", wheel_r, 0.06, "frame", -1.00, 0, -0.68),
    ]

def _train_model(color):
    wheel_r = 0.55
    wheel_w = 0.30
    return [
        _part("engine", "cube", {"dx": 3.6, "dy": 1.9, "dz": 1.5}, color),
        _part("cab", "cube", {"dx": 1.3, "dy": 1.85, "dz": 1.5}, color,
              "engine", position="center", offset_x=-2.30, offset_z=0.40),
        _part("cab_window", "cube", {"dx": 0.06, "dy": 1.60, "dz": 0.65},
              "lightblue", "cab", position="center",
              offset_x=-0.65, offset_z=0.45),
        _part("smokestack", "cylinder", {"radius": 0.22, "height": 0.9},
              "black", "engine", position="top", offset_x=1.10),
        _part("cowcatcher", "pyramid", {"base": 1.85, "height": 0.85},
              "black", "engine", position="right", rotation={"ry": 90}),
        _part("headlight", "sphere", {"radius": 0.14}, "yellow",
              "engine", position="right", offset_z=0.55),
        _wheel("wheel_fl", wheel_r, wheel_w, "engine", 1.00, -1.0, -1.05),
        _wheel("wheel_fr", wheel_r, wheel_w, "engine", 1.00, 1.0, -1.05),
        _wheel("wheel_bl", wheel_r, wheel_w, "engine", -1.00, -1.0, -1.05),
        _wheel("wheel_br", wheel_r, wheel_w, "engine", -1.00, 1.0, -1.05),
    ]

def _plane_model(color):
    return [
        _part("fuselage", "cylinder", {"radius": 0.48, "height": 6.2},
              color, rotation={"ry": 90}),
        _part("nose", "cone", {"radius": 0.48, "height": 0.9}, color,
              "fuselage", position="right", rotation={"ry": 90}),
        _part("tail_cone", "cone", {"radius": 0.48, "height": 0.8}, color,
              "fuselage", position="left", rotation={"ry": -90}),
        _part("main_wings", "cube", {"dx": 1.3, "dy": 7.4, "dz": 0.14},
              color, "fuselage", position="center", offset_x=0.35),
        _part("tail_wing", "cube", {"dx": 0.75, "dy": 2.6, "dz": 0.10},
              color, "fuselage", position="center", offset_x=-2.55),
        _part("tail_fin", "cube", {"dx": 0.70, "dy": 0.12, "dz": 1.25},
              color, "fuselage", position="center",
              offset_x=-2.55, offset_z=0.75),
        _part("cockpit", "sphere", {"radius": 0.26}, "lightblue",
              "fuselage", position="center", offset_x=1.40, offset_z=0.30),
        _part("engine_left", "cylinder",
              {"radius": 0.20, "height": 0.75}, "silver",
              "main_wings", position="bottom", offset_y=1.4),
        _part("engine_right", "cylinder",
              {"radius": 0.20, "height": 0.75}, "silver",
              "main_wings", position="bottom", offset_y=-1.4),
    ]

def _helicopter_model(color):
    return [
        _part("body", "cube", {"dx": 2.6, "dy": 1.15, "dz": 1.05}, color),
        _part("cockpit", "sphere", {"radius": 0.62}, "lightblue",
              "body", position="center", offset_x=1.20, offset_z=0.05),
        _part("tail_boom", "cube", {"dx": 2.4, "dy": 0.24, "dz": 0.24},
              color, "body", position="center", offset_x=-2.20, offset_z=0.15),
        _part("tail_fin", "cube", {"dx": 0.42, "dy": 0.12, "dz": 0.70},
              color, "tail_boom", position="center",
              offset_x=-1.00, offset_z=0.42),
        _part("rotor_mast", "cylinder", {"radius": 0.14, "height": 0.30},
              "darkgray", "body", position="top"),
        _part("main_rotor", "cube", {"dx": 0.16, "dy": 5.2, "dz": 0.08},
              "darkgray", "rotor_mast", position="top"),
        _part("skid_left", "cube", {"dx": 2.6, "dy": 0.12, "dz": 0.12},
              "black", "body", position="bottom", offset_y=-0.72),
        _part("skid_right", "cube", {"dx": 2.6, "dy": 0.12, "dz": 0.12},
              "black", "body", position="bottom", offset_y=0.72),
    ]

def _rocket_model(color):
    return [
        _part("body", "cylinder", {"radius": 0.75, "height": 3.6}, color),
        _part("nose", "cone", {"radius": 0.75, "height": 1.5}, "white",
              "body", position="top"),
        _part("porthole", "cylinder", {"radius": 0.20, "height": 0.05},
              "lightblue", "body", position="front", offset_z=0.9),
        _part("fin_left", "cube", {"dx": 0.65, "dy": 0.12, "dz": 0.95},
              "red", "body", position="left", offset_z=-1.40),
        _part("fin_right", "cube", {"dx": 0.65, "dy": 0.12, "dz": 0.95},
              "red", "body", position="right", offset_z=-1.40),
        _part("fin_back", "cube", {"dx": 0.12, "dy": 0.65, "dz": 0.95},
              "red", "body", position="back", offset_z=-1.40),
        _part("nozzle", "cone", {"radius": 0.55, "height": 0.35}, "darkgray",
              "body", position="bottom", rotation={"rx": 180}),
    ]

def _boat_model(color):
    return [
        _part("hull", "cube", {"dx": 4.2, "dy": 1.75, "dz": 0.65}, color),
        _part("bow", "cone", {"radius": 0.88, "height": 1.0}, color,
              "hull", position="right", rotation={"ry": 90}),
        _part("deck", "cube", {"dx": 3.4, "dy": 1.6, "dz": 0.10}, "tan",
              "hull", position="top"),
        _part("cabin", "cube", {"dx": 1.25, "dy": 1.15, "dz": 0.70},
              "white", "deck", position="top", offset_x=-0.4),
        _part("cabin_window", "cube", {"dx": 1.05, "dy": 0.04, "dz": 0.32},
              "lightblue", "cabin", position="front", offset_z=0.15),
        _part("mast", "cylinder", {"radius": 0.06, "height": 2.3}, "gray",
              "deck", position="top", offset_x=1.0),
        _part("sail", "cube", {"dx": 0.05, "dy": 0.95, "dz": 1.4},
              "white", "mast", position="center",
              offset_z=0.15, offset_y=0.52),
    ]

# Furniture
def _table_model(color):
    top_dz = 0.22
    leg_h = 2.4
    return [
        _part("top", "cube", {"dx": 4.4, "dy": 2.4, "dz": top_dz}, color),
        _part("apron_front", "cube", {"dx": 4.0, "dy": 0.10, "dz": 0.35},
              color, "top", position="bottom", offset_y=1.05),
        _part("apron_back", "cube", {"dx": 4.0, "dy": 0.10, "dz": 0.35},
              color, "top", position="bottom", offset_y=-1.05),
        _part("leg_fl", "cylinder", {"radius": 0.13, "height": leg_h},
              "brown", "top", position="bottom", offset_x=1.70, offset_y=0.85),
        _part("leg_fr", "cylinder", {"radius": 0.13, "height": leg_h},
              "brown", "top", position="bottom", offset_x=1.70, offset_y=-0.85),
        _part("leg_bl", "cylinder", {"radius": 0.13, "height": leg_h},
              "brown", "top", position="bottom", offset_x=-1.70, offset_y=0.85),
        _part("leg_br", "cylinder", {"radius": 0.13, "height": leg_h},
              "brown", "top", position="bottom", offset_x=-1.70, offset_y=-0.85),
    ]

def _bench_model(color):
    return [
        _part("seat", "cube", {"dx": 3.8, "dy": 0.70, "dz": 0.22}, color),
        _part("back", "cube", {"dx": 3.8, "dy": 0.12, "dz": 1.05}, color,
              "seat", position="back", offset_z=0.60),
        _part("leg_left", "cube", {"dx": 0.18, "dy": 0.60, "dz": 1.55},
              "black", "seat", position="bottom", offset_x=-1.55),
        _part("leg_right", "cube", {"dx": 0.18, "dy": 0.60, "dz": 1.55},
              "black", "seat", position="bottom", offset_x=1.55),
    ]

def _chair_model(color):
    seat_dz = 0.22
    leg_h = 1.7
    return [
        _part("seat", "cube", {"dx": 1.55, "dy": 1.55, "dz": seat_dz}, color),
        _part("back", "cube", {"dx": 1.55, "dy": 0.14, "dz": 1.55}, color,
              "seat", position="back", offset_z=0.85),
        _part("leg_fl", "cylinder", {"radius": 0.07, "height": leg_h},
              "brown", "seat", position="bottom",
              offset_x=0.62, offset_y=0.62),
        _part("leg_fr", "cylinder", {"radius": 0.07, "height": leg_h},
              "brown", "seat", position="bottom",
              offset_x=0.62, offset_y=-0.62),
        _part("leg_bl", "cylinder", {"radius": 0.07, "height": leg_h},
              "brown", "seat", position="bottom",
              offset_x=-0.62, offset_y=0.62),
        _part("leg_br", "cylinder", {"radius": 0.07, "height": leg_h},
              "brown", "seat", position="bottom",
              offset_x=-0.62, offset_y=-0.62),
    ]

def _sofa_model(color):
    return [
        _part("base", "cube", {"dx": 3.4, "dy": 1.4, "dz": 0.50}, color),
        _part("cushion_left", "cube", {"dx": 1.5, "dy": 1.25, "dz": 0.22},
              color, "base", position="top", offset_x=-0.85),
        _part("cushion_right", "cube", {"dx": 1.5, "dy": 1.25, "dz": 0.22},
              color, "base", position="top", offset_x=0.85),
        _part("back", "cube", {"dx": 3.4, "dy": 0.28, "dz": 1.35}, color,
              "base", position="back", offset_z=0.70),
        _part("arm_left", "cube", {"dx": 0.30, "dy": 1.4, "dz": 1.00}, color,
              "base", position="left", offset_z=0.28),
        _part("arm_right", "cube", {"dx": 0.30, "dy": 1.4, "dz": 1.00}, color,
              "base", position="right", offset_z=0.28),
    ]

def _bed_model(color):
    return [
        _part("mattress", "cube", {"dx": 3.8, "dy": 2.2, "dz": 0.45}, color),
        _part("headboard", "cube", {"dx": 0.22, "dy": 2.2, "dz": 1.60},
              "brown", "mattress", position="left", offset_z=0.60),
        _part("footboard", "cube", {"dx": 0.22, "dy": 2.2, "dz": 0.85},
              "brown", "mattress", position="right", offset_z=0.25),
        _part("pillow_left", "cube", {"dx": 0.60, "dy": 0.85, "dz": 0.16},
              "white", "mattress", position="top",
              offset_x=-1.35, offset_y=0.55),
        _part("pillow_right", "cube", {"dx": 0.60, "dy": 0.85, "dz": 0.16},
              "white", "mattress", position="top",
              offset_x=-1.35, offset_y=-0.55),
        _part("blanket", "cube", {"dx": 2.0, "dy": 2.2, "dz": 0.12},
              "red", "mattress", position="top", offset_x=0.65),
    ]

def _bookshelf_model(color):
    return [
        _part("frame", "cube", {"dx": 0.35, "dy": 2.0, "dz": 3.0}, color),
        _part("side_top", "cube", {"dx": 1.6, "dy": 0.14, "dz": 0.14},
              color, "frame", position="top", offset_x=0.90),
        _part("side_bottom", "cube", {"dx": 1.6, "dy": 0.14, "dz": 0.14},
              color, "frame", position="bottom", offset_x=0.90),
        _part("shelf_mid", "cube", {"dx": 1.6, "dy": 0.14, "dz": 0.14},
              color, "frame", position="center", offset_x=0.90),
        _part("book_a", "cube", {"dx": 0.32, "dy": 0.10, "dz": 0.68},
              "red", "shelf_mid", position="top", offset_x=-0.05),
        _part("book_b", "cube", {"dx": 0.28, "dy": 0.10, "dz": 0.60},
              "blue", "shelf_mid", position="top", offset_x=0.32),
        _part("book_c", "cube", {"dx": 0.26, "dy": 0.10, "dz": 0.62},
              "green", "shelf_mid", position="top", offset_x=0.66),
    ]

def _lamp_model(color):
    return [
        _part("base", "cylinder", {"radius": 0.60, "height": 0.16}, "gray"),
        _part("stem", "cylinder", {"radius": 0.08, "height": 2.2}, "gold",
              "base", position="top"),
        _part("shade", "cone", {"radius": 0.72, "height": 0.85}, color,
              "stem", position="top"),
    ]

# Devices and appliances
def _device_model(kind, color):
    if kind == "mouse":
        return [
            _part("shell", "cube", {"dx": 1.35, "dy": 2.1, "dz": 0.52}, color),
            _part("left_button", "cube", {"dx": 0.60, "dy": 0.82, "dz": 0.06},
                  color, "shell", position="top",
                  offset_x=-0.35, offset_y=-0.40),
            _part("right_button", "cube",
                  {"dx": 0.60, "dy": 0.82, "dz": 0.06}, color, "shell",
                  position="top", offset_x=0.35, offset_y=-0.40),
            _part("scroll_wheel", "cylinder",
                  {"radius": 0.11, "height": 0.14}, "black", "shell",
                  position="top", rotation={"rx": 90}, offset_y=-0.42),
            _part("base", "cube", {"dx": 1.28, "dy": 2.0, "dz": 0.06},
                  "black", "shell", position="bottom"),
        ]
    if kind == "cell_phone":
        return [
            _part("body", "cube", {"dx": 1.5, "dy": 0.26, "dz": 3.0}, color),
            _part("screen", "cube", {"dx": 1.24, "dy": 0.03, "dz": 2.50},
                  "black", "body", position="front"),
            _part("camera", "cylinder", {"radius": 0.13, "height": 0.04},
                  "blue", "body", position="back",
                  rotation={"rx": 90}, offset_x=-0.42, offset_z=1.20),
        ]
    if kind == "laptop":
        return [
            _part("base", "cube", {"dx": 3.4, "dy": 2.35, "dz": 0.20}, color),
            _part("keyboard", "cube", {"dx": 3.0, "dy": 1.5, "dz": 0.03},
                  "darkgray", "base", position="top", offset_y=-0.25),
            _part("touchpad", "cube", {"dx": 0.9, "dy": 0.65, "dz": 0.02},
                  "gray", "base", position="top", offset_y=0.75),
            _part("display", "cube", {"dx": 3.3, "dy": 0.18, "dz": 2.15},
                  "black", "base", position="back",
                  rotation={"rx": -12}, offset_z=1.05),
            _part("screen", "cube", {"dx": 2.85, "dy": 0.03, "dz": 1.70},
                  "lightblue", "display", position="front"),
        ]
    if kind == "keyboard":
        return [
            _part("body", "cube", {"dx": 4.0, "dy": 1.5, "dz": 0.16}, color),
            _part("keys", "cube", {"dx": 3.55, "dy": 1.10, "dz": 0.05},
                  "white", "body", position="top"),
        ]
    return [
        _part("body", "cube", {"dx": 0.78, "dy": 2.15, "dz": 0.22}, color),
        _part("panel", "cube", {"dx": 0.55, "dy": 1.55, "dz": 0.03},
              "black", "body", position="top"),
        _part("power_button", "sphere", {"radius": 0.06}, "red",
              "panel", position="top", offset_y=0.95),
    ]

def _appliance_model(kind, color):
    if kind == "clock":
        return [
            _part("body", "cylinder", {"radius": 1.0, "height": 0.22}, color,
                  rotation={"rx": 90}),
            _part("face", "cylinder", {"radius": 0.82, "height": 0.03},
                  "white", "body", position="front", rotation={"rx": 90}),
            _part("hand_hour", "cube", {"dx": 0.05, "dy": 0.05, "dz": 0.42},
                  "black", "face", position="center", offset_z=0.20),
            _part("hand_minute", "cube", {"dx": 0.05, "dy": 0.05, "dz": 0.58},
                  "black", "face", position="center", rotation={"ry": 90},
                  offset_x=0.24),
        ]
    if kind == "tv":
        return [
            _part("body", "cube", {"dx": 3.4, "dy": 0.22, "dz": 2.0}, color),
            _part("screen", "cube", {"dx": 3.05, "dy": 0.03, "dz": 1.72},
                  "black", "body", position="front"),
            _part("stand", "cube", {"dx": 0.85, "dy": 0.45, "dz": 0.14},
                  "black", "body", position="bottom"),
        ]
    if kind in {"microwave", "oven"}:
        return [
            _part("body", "cube", {"dx": 2.8, "dy": 1.4, "dz": 1.6}, color),
            _part("door", "cube", {"dx": 1.9, "dy": 0.05, "dz": 1.2},
                  "black", "body", position="front", offset_x=-0.30),
            _part("window", "cube", {"dx": 1.4, "dy": 0.02, "dz": 0.75},
                  "darkgray", "door", position="front"),
            _part("handle", "cube", {"dx": 0.08, "dy": 0.08, "dz": 0.95},
                  "silver", "door", position="right"),
            _part("panel", "cube", {"dx": 0.55, "dy": 0.05, "dz": 1.2},
                  "darkgray", "body", position="front", offset_x=1.05),
        ]
    if kind == "toaster":
        return [
            _part("body", "cube", {"dx": 1.8, "dy": 1.1, "dz": 1.0}, color),
            _part("slot_l", "cube", {"dx": 0.55, "dy": 0.14, "dz": 0.04},
                  "black", "body", position="top", offset_x=-0.40),
            _part("slot_r", "cube", {"dx": 0.55, "dy": 0.14, "dz": 0.04},
                  "black", "body", position="top", offset_x=0.40),
            _part("lever", "cube", {"dx": 0.08, "dy": 0.10, "dz": 0.20},
                  "black", "body", position="front", offset_x=0.70),
        ]
    if kind == "refrigerator":
        return [
            _part("body", "cube", {"dx": 1.8, "dy": 1.4, "dz": 4.0}, color),
            _part("freezer_seam", "cube",
                  {"dx": 1.82, "dy": 1.42, "dz": 0.03}, "darkgray",
                  "body", position="center", offset_z=0.95),
            _part("handle_top", "cylinder",
                  {"radius": 0.045, "height": 0.65}, "silver",
                  "body", position="front", offset_x=0.68, offset_z=1.40),
            _part("handle_bottom", "cylinder",
                  {"radius": 0.045, "height": 1.40}, "silver",
                  "body", position="front", offset_x=0.68, offset_z=-0.40),
        ]
    if kind == "toilet":
        return [
            _part("bowl", "cylinder", {"radius": 0.62, "height": 0.70},
                  "white"),
            _part("seat", "cylinder", {"radius": 0.70, "height": 0.07},
                  "white", "bowl", position="top"),
            _part("tank", "cube", {"dx": 0.85, "dy": 0.32, "dz": 0.90},
                  "white", "bowl", position="back", offset_z=0.40),
        ]
    return [
        _part("basin", "cylinder", {"radius": 0.72, "height": 0.45}, "white"),
        _part("counter", "cube", {"dx": 1.9, "dy": 1.3, "dz": 0.10}, color,
              "basin", position="top"),
        _part("faucet", "cylinder", {"radius": 0.05, "height": 0.55},
              "silver", "counter", position="top", offset_y=0.40),
    ]

# Containers, utensils, writing implements
def _container_model(kind, color):
    if kind == "bottle":
        return [
            _part("body", "cylinder", {"radius": 0.52, "height": 1.8}, color),
            _part("shoulder", "cone", {"radius": 0.52, "height": 0.4}, color,
                  "body", position="top"),
            _part("neck", "cylinder", {"radius": 0.20, "height": 0.45}, color,
                  "shoulder", position="top"),
            _part("cap", "cylinder", {"radius": 0.24, "height": 0.14}, "black",
                  "neck", position="top"),
        ]
    if kind == "vase":
        return [
            _part("body", "cylinder", {"radius": 0.55, "height": 1.2}, color),
            _part("shoulder", "sphere", {"radius": 0.58}, color, "body",
                  position="top", offset_z=-0.10),
            _part("neck", "cylinder", {"radius": 0.28, "height": 0.65}, color,
                  "shoulder", position="top"),
        ]
    if kind == "book":
        return [
            _part("cover", "cube", {"dx": 2.3, "dy": 0.68, "dz": 3.0}, color),
            _part("pages", "cube", {"dx": 2.15, "dy": 0.60, "dz": 2.85},
                  "white", "cover", position="right"),
            _part("spine", "cube", {"dx": 0.14, "dy": 0.70, "dz": 3.0},
                  "brown", "cover", position="left"),
        ]
    if kind == "suitcase":
        return [
            _part("body", "cube", {"dx": 2.2, "dy": 0.80, "dz": 3.0}, color),
            _part("trim", "cube", {"dx": 2.22, "dy": 0.82, "dz": 0.10},
                  "black", "body", position="center", offset_z=0.90),
            _part("handle", "cylinder", {"radius": 0.06, "height": 0.55},
                  "black", "body", position="top"),
        ]
    if kind == "backpack":
        return [
            _part("body", "cube", {"dx": 2.0, "dy": 1.0, "dz": 2.6}, color),
            _part("top_flap", "cube", {"dx": 2.02, "dy": 1.02, "dz": 0.35},
                  color, "body", position="top"),
            _part("strap_left", "cube", {"dx": 0.10, "dy": 0.10, "dz": 1.6},
                  "black", "body", position="back", offset_x=-0.55),
            _part("strap_right", "cube", {"dx": 0.10, "dy": 0.10, "dz": 1.6},
                  "black", "body", position="back", offset_x=0.55),
        ]
    if kind == "cup":
        return [
            _part("body", "cylinder", {"radius": 0.70, "height": 1.4}, color),
            _part("rim", "cylinder", {"radius": 0.74, "height": 0.06},
                  "white", "body", position="top"),
            _part("handle", "cylinder", {"radius": 0.06, "height": 0.70},
                  color, "body", position="right",
                  rotation={"rx": 90}, offset_z=0.10),
        ]
    return [
        _part("body", "cylinder", {"radius": 0.90, "height": 0.55}, color),
        _part("rim", "cylinder", {"radius": 0.96, "height": 0.08}, "white",
              "body", position="top"),
    ]

def _utensil_model(kind, color):
    if kind == "knife":
        return [
            _part("handle", "cube", {"dx": 0.26, "dy": 2.0, "dz": 0.14},
                  color),
            _part("blade", "cube", {"dx": 0.42, "dy": 2.2, "dz": 0.05},
                  "silver", "handle", position="front"),
        ]
    if kind == "spoon":
        return [
            _part("handle", "cube", {"dx": 0.22, "dy": 2.6, "dz": 0.10},
                  color),
            _part("bowl", "sphere", {"radius": 0.48}, "silver",
                  "handle", position="front", offset_z=0.05),
        ]
    if kind == "fork":
        parts = [
            _part("handle", "cube", {"dx": 0.22, "dy": 2.2, "dz": 0.10},
                  color),
            _part("head", "cube", {"dx": 0.55, "dy": 0.7, "dz": 0.06},
                  "silver", "handle", position="front"),
        ]
        for index, x in enumerate((-0.18, 0, 0.18)):
            parts.append(_part(
                f"tine_{index}", "cube",
                {"dx": 0.07, "dy": 0.6, "dz": 0.06}, "silver",
                "head", position="front", offset_x=x,
            ))
        return parts
    if kind == "scissors":
        return [
            _part("pivot", "cylinder", {"radius": 0.14, "height": 0.22},
                  "gold"),
            _part("blade_left", "cube", {"dx": 2.6, "dy": 0.10, "dz": 0.08},
                  "silver", "pivot", position="center",
                  rotation={"rz": 18}, offset_x=1.30),
            _part("blade_right", "cube", {"dx": 2.6, "dy": 0.10, "dz": 0.08},
                  "silver", "pivot", position="center",
                  rotation={"rz": -18}, offset_x=1.30),
            _part("handle_left", "cylinder",
                  {"radius": 0.48, "height": 0.16}, color, "pivot",
                  position="center", rotation={"rx": 90},
                  offset_x=-0.90, offset_y=0.30),
            _part("handle_right", "cylinder",
                  {"radius": 0.48, "height": 0.16}, color, "pivot",
                  position="center", rotation={"rx": 90},
                  offset_x=-0.90, offset_y=-0.30),
        ]
    return [
        _part("handle", "cube", {"dx": 0.18, "dy": 2.8, "dz": 0.14}, color),
        _part("head", "cube", {"dx": 0.44, "dy": 0.65, "dz": 0.16},
              "white", "handle", position="front"),
        _part("bristles", "cube", {"dx": 0.36, "dy": 0.42, "dz": 0.20},
              "white", "head", position="front"),
    ]

def _writing_model(kind, color):
    if kind == "pencil":
        return [
            _part("body", "cylinder", {"radius": 0.24, "height": 4.4},
                  color, rotation={"ry": 90}),
            _part("tip", "cone", {"radius": 0.24, "height": 0.7}, "tan",
                  "body", position="right", rotation={"ry": 90}),
            _part("graphite", "cone", {"radius": 0.07, "height": 0.25},
                  "black", "tip", position="right", rotation={"ry": 90}),
            _part("ferrule", "cylinder", {"radius": 0.26, "height": 0.4},
                  "silver", "body", position="left", rotation={"ry": 90}),
            _part("eraser", "cylinder", {"radius": 0.25, "height": 0.4},
                  "pink", "ferrule", position="left", rotation={"ry": 90}),
        ]
    if kind == "eraser":
        return [
            _part("body", "cube", {"dx": 1.8, "dy": 0.9, "dz": 0.55},
                  color),
            _part("band", "cube", {"dx": 0.32, "dy": 0.92, "dz": 0.58},
                  "white", "body", position="center", offset_x=-0.45),
        ]
    parts = [
        _part("body", "cube", {"dx": 5.5, "dy": 0.5, "dz": 0.10}, color)
    ]
    for index in range(1, 6):
        parts.append(_part(
            f"mark_{index}", "cube",
            {"dx": 0.03, "dy": 0.10, "dz": 0.04}, "black",
            "body", position="top",
            offset_x=-2.75 + index, offset_y=-0.18,
        ))
    return parts

# Food
def _food_model(kind, color):
    if kind == "apple":
        return [
            _part("fruit", "sphere", {"radius": 0.78}, color),
            _part("stem", "cylinder", {"radius": 0.06, "height": 0.28},
                  "brown", "fruit", position="top"),
            _part("leaf", "cube", {"dx": 0.36, "dy": 0.10, "dz": 0.05},
                  "green", "stem", position="top",
                  rotation={"rz": -25}, offset_x=0.20),
        ]
    if kind == "orange":
        return [
            _part("fruit", "sphere", {"radius": 0.76}, "orange"),
            _part("stem", "cylinder", {"radius": 0.05, "height": 0.14},
                  "green", "fruit", position="top"),
        ]
    if kind == "banana":
        return [
            _part("body", "cylinder", {"radius": 0.20, "height": 2.4},
                  "yellow", rotation={"ry": 65}),
            _part("tip_left", "sphere", {"radius": 0.20}, "yellow",
                  "body", position="left"),
            _part("tip_right", "sphere", {"radius": 0.20}, "yellow",
                  "body", position="right"),
            _part("stem", "cylinder", {"radius": 0.08, "height": 0.22},
                  "brown", "body", position="right"),
        ]
    if kind == "donut":
        return [
            _part("cake", "cylinder", {"radius": 0.90, "height": 0.32},
                  color),
            _part("icing", "cylinder", {"radius": 0.92, "height": 0.08},
                  "pink", "cake", position="top"),
            _part("hole", "cylinder", {"radius": 0.26, "height": 0.50},
                  "black", "cake", position="center"),
        ]
    if kind == "pizza":
        parts = [
            _part("crust", "cylinder", {"radius": 1.35, "height": 0.20},
                  "tan"),
            _part("sauce", "cylinder", {"radius": 1.22, "height": 0.05},
                  "red", "crust", position="top"),
        ]
        for index, (x, y) in enumerate(((0.45, 0.45), (-0.45, 0.35),
                                        (0.0, -0.55), (0.55, -0.35),
                                        (-0.55, -0.30))):
            parts.append(_part(
                f"pepperoni_{index}", "cylinder",
                {"radius": 0.18, "height": 0.05}, "red",
                "sauce", position="top", offset_x=x, offset_y=y,
            ))
        return parts
    if kind == "cake":
        return [
            _part("body", "cylinder", {"radius": 1.0, "height": 1.0}, color),
            _part("icing", "cylinder", {"radius": 1.02, "height": 0.14},
                  "pink", "body", position="top"),
            _part("candle", "cylinder", {"radius": 0.05, "height": 0.45},
                  "white", "icing", position="top"),
            _part("flame", "sphere", {"radius": 0.07}, "orange",
                  "candle", position="top"),
        ]
    if kind == "sandwich":
        return [
            _part("bread_bottom", "cube",
                  {"dx": 2.0, "dy": 1.6, "dz": 0.22}, "tan"),
            _part("filling", "cube", {"dx": 1.9, "dy": 1.5, "dz": 0.32},
                  color, "bread_bottom", position="top"),
            _part("bread_top", "cube", {"dx": 2.0, "dy": 1.6, "dz": 0.22},
                  "tan", "filling", position="top"),
        ]
    if kind == "hot_dog":
        return [
            _part("bun", "cube", {"dx": 2.5, "dy": 0.85, "dz": 0.45},
                  "tan"),
            _part("sausage", "cylinder", {"radius": 0.24, "height": 2.5},
                  "brown", "bun", position="top", rotation={"ry": 90}),
            _part("mustard", "cube", {"dx": 2.2, "dy": 0.08, "dz": 0.03},
                  "yellow", "sausage", position="top"),
        ]
    if kind == "broccoli":
        return [
            _part("stem", "cylinder", {"radius": 0.20, "height": 1.0},
                  "green"),
            _part("crown", "sphere", {"radius": 0.85}, "darkgreen",
                  "stem", position="top"),
            _part("floret_left", "sphere", {"radius": 0.32}, "darkgreen",
                  "crown", position="center", offset_x=-0.50),
            _part("floret_right", "sphere", {"radius": 0.32}, "darkgreen",
                  "crown", position="center", offset_x=0.50),
        ]
    return [
        _part("root", "cone", {"radius": 0.42, "height": 1.8}, "orange",
              rotation={"rz": 180}),
        _part("leaves", "cylinder", {"radius": 0.20, "height": 0.8},
              "green", "root", position="top"),
    ]

# Sports
def _sports_model(kind, color):
    if kind == "sports_ball":
        return [
            _part("ball", "sphere", {"radius": 0.90}, color),
            _part("stripe_h", "cylinder", {"radius": 0.92, "height": 0.05},
                  "white", "ball", position="center", rotation={"rx": 90}),
            _part("stripe_v", "cylinder", {"radius": 0.92, "height": 0.05},
                  "white", "ball", position="center"),
        ]
    if kind == "frisbee":
        return [
            _part("disc", "cylinder", {"radius": 1.0, "height": 0.15},
                  color),
            _part("rim", "cylinder", {"radius": 0.86, "height": 0.06},
                  "white", "disc", position="top"),
        ]
    if kind == "skis":
        return [
            _part("ski_left", "cube", {"dx": 0.34, "dy": 5.0, "dz": 0.10},
                  color, position="center", offset_x=-0.32),
            _part("ski_right", "cube", {"dx": 0.34, "dy": 5.0, "dz": 0.10},
                  color, position="center", offset_x=0.32),
            _part("binding_left", "cube",
                  {"dx": 0.55, "dy": 0.40, "dz": 0.10}, "black",
                  "ski_left", position="top"),
            _part("binding_right", "cube",
                  {"dx": 0.55, "dy": 0.40, "dz": 0.10}, "black",
                  "ski_right", position="top"),
        ]
    if kind == "skateboard":
        length = 4.4
        parts = [
            _part("board", "cube", {"dx": 0.65, "dy": length, "dz": 0.10},
                  color),
        ]
        for side, sign in (("front", 1), ("rear", -1)):
            truck_y = sign * length / 3
            parts.append(_part(
                f"truck_{side}", "cylinder",
                {"radius": 0.14, "height": 0.65}, "silver",
                "board", position="bottom",
                rotation={"rx": 90}, offset_y=truck_y,
            ))
        return parts
    if kind in {"snowboard", "surfboard"}:
        length = 5.0 if kind == "snowboard" else 5.4
        parts = [
            _part("board", "cube",
                  {"dx": 0.85, "dy": length, "dz": 0.10}, color)
        ]
        if kind == "snowboard":
            parts += [
                _part("binding_front", "cube",
                      {"dx": 0.55, "dy": 0.45, "dz": 0.12}, "black",
                      "board", position="top", offset_y=0.80),
                _part("binding_back", "cube",
                      {"dx": 0.55, "dy": 0.45, "dz": 0.12}, "black",
                      "board", position="top", offset_y=-0.80),
            ]
        else:
            parts.append(_part(
                "fin", "cube", {"dx": 0.10, "dy": 0.55, "dz": 0.45},
                "white", "board", position="bottom", offset_y=-length / 3,
            ))
        return parts
    if kind == "baseball_bat":
        return [
            _part("handle", "cylinder", {"radius": 0.14, "height": 1.5},
                  color, rotation={"ry": 90}),
            _part("barrel", "cylinder", {"radius": 0.30, "height": 1.8},
                  color, "handle", position="right", rotation={"ry": 90}),
            _part("cap", "sphere", {"radius": 0.30}, color, "barrel",
                  position="right"),
        ]
    if kind == "tennis_racket":
        return [
            _part("handle", "cylinder", {"radius": 0.12, "height": 1.1},
                  color, rotation={"ry": 90}),
            _part("rim", "cylinder", {"radius": 0.65, "height": 0.08},
                  "silver", "handle", position="right",
                  rotation={"rx": 90}),
            _part("strings", "cylinder", {"radius": 0.60, "height": 0.02},
                  "white", "rim", position="center",
                  rotation={"rx": 90}),
        ]
    if kind == "baseball_glove":
        return [
            _part("palm", "sphere", {"radius": 0.72}, color),
            _part("thumb", "cylinder", {"radius": 0.16, "height": 0.75},
                  color, "palm", position="left", rotation={"ry": 55}),
            _part("fingers", "cylinder", {"radius": 0.28, "height": 0.75},
                  color, "palm", position="right", rotation={"ry": 90}),
        ]
    return [
        _part("sail", "pyramid", {"base": 1.8, "height": 0.10}, color,
              rotation={"rx": 90}),
        _part("spine", "cube", {"dx": 0.05, "dy": 0.05, "dz": 2.3},
              "brown", "sail", position="center"),
        _part("tail", "cube", {"dx": 0.08, "dy": 1.7, "dz": 0.05}, "red",
              "sail", position="bottom"),
    ]

# Accessories
def _accessory_model(kind, color):
    if kind == "traffic_light":
        return [
            _part("post", "cylinder", {"radius": 0.11, "height": 2.8},
                  "gray"),
            _part("housing", "cube", {"dx": 0.72, "dy": 0.48, "dz": 1.75},
                  "black", "post", position="top"),
            _part("red_lamp", "sphere", {"radius": 0.16}, "red",
                  "housing", position="center", offset_z=0.55),
            _part("yellow_lamp", "sphere", {"radius": 0.16}, "yellow",
                  "housing", position="center"),
            _part("green_lamp", "sphere", {"radius": 0.16}, "green",
                  "housing", position="center", offset_z=-0.55),
        ]
    if kind == "fire_hydrant":
        return [
            _part("body", "cylinder", {"radius": 0.40, "height": 1.4},
                  "red"),
            _part("cap", "cylinder", {"radius": 0.50, "height": 0.16},
                  "red", "body", position="top"),
            _part("valve_left", "cylinder",
                  {"radius": 0.20, "height": 0.26}, "silver",
                  "body", position="left", offset_z=0.20),
            _part("valve_right", "cylinder",
                  {"radius": 0.20, "height": 0.26}, "silver",
                  "body", position="right", offset_z=0.20),
        ]
    if kind == "stop_sign":
        return [
            _part("post", "cylinder", {"radius": 0.09, "height": 2.6},
                  "gray"),
            _part("sign", "cylinder", {"radius": 0.72, "height": 0.10},
                  "red", "post", position="top", rotation={"rx": 90}),
            _part("face", "cylinder", {"radius": 0.55, "height": 0.02},
                  "white", "sign", position="front", rotation={"rx": 90}),
        ]
    if kind == "parking_meter":
        return [
            _part("post", "cylinder", {"radius": 0.09, "height": 2.2},
                  "gray"),
            _part("head", "cube", {"dx": 0.68, "dy": 0.42, "dz": 0.85},
                  "gray", "post", position="top"),
            _part("screen", "cube", {"dx": 0.40, "dy": 0.03, "dz": 0.26},
                  "black", "head", position="front"),
        ]
    if kind == "umbrella":
        return [
            _part("shaft", "cylinder", {"radius": 0.06, "height": 2.8},
                  "black"),
            _part("canopy", "cone", {"radius": 1.25, "height": 0.70},
                  color, "shaft", position="top", rotation={"rz": 180}),
            _part("handle", "sphere", {"radius": 0.18}, "black", "shaft",
                  position="bottom"),
        ]
    if kind == "tie":
        return [
            _part("knot", "cube", {"dx": 0.32, "dy": 0.16, "dz": 0.32},
                  color),
            _part("blade", "pyramid", {"base": 0.50, "height": 1.7},
                  color, "knot", position="bottom", rotation={"rx": 180}),
        ]
    if kind == "handbag":
        return [
            _part("bag", "cube", {"dx": 1.8, "dy": 0.75, "dz": 1.4},
                  color),
            _part("handle_left", "cylinder",
                  {"radius": 0.05, "height": 0.75}, "black",
                  "bag", position="top", rotation={"ry": 90},
                  offset_x=-0.55),
            _part("handle_right", "cylinder",
                  {"radius": 0.05, "height": 0.75}, "black",
                  "bag", position="top", rotation={"ry": 90},
                  offset_x=0.55),
        ]
    if kind == "potted_plant":
        return [
            _part("pot", "cone", {"radius": 0.72, "height": 0.85}, "brown",
                  rotation={"rz": 180}),
            _part("soil", "cylinder", {"radius": 0.65, "height": 0.07},
                  "saddlebrown", "pot", position="top"),
            _part("stem", "cylinder", {"radius": 0.08, "height": 1.4},
                  "green", "soil", position="top"),
            _part("leaves", "sphere", {"radius": 0.78}, "green", "stem",
                  position="top"),
        ]
    return [
        _part("body", "cube", {"dx": 1.15, "dy": 1.35, "dz": 0.75}, color),
        _part("nozzle", "cylinder", {"radius": 0.32, "height": 0.75},
              "black", "body", position="front", rotation={"rx": 90}),
        _part("handle", "cube", {"dx": 0.42, "dy": 0.42, "dz": 1.25},
              color, "body", position="bottom"),
    ]

# Nature, people, animals
def _nature_model(kind, color):
    if kind == "tree":
        return [
            _part("trunk", "cylinder", {"radius": 0.30, "height": 2.4},
                  "brown"),
            _part("crown_low", "sphere", {"radius": 1.35}, "green", "trunk",
                  position="top"),
            _part("crown_high", "sphere", {"radius": 0.95}, "green",
                  "crown_low", position="top", offset_y=0.30),
        ]
    parts = [
        _part("stem", "cylinder", {"radius": 0.08, "height": 2.0},
              "green"),
        _part("leaf", "cube", {"dx": 0.65, "dy": 0.10, "dz": 0.32},
              "green", "stem", position="right", offset_z=-0.40),
        _part("center", "sphere", {"radius": 0.20}, "yellow", "stem",
              position="top"),
    ]
    for index, angle in enumerate((0, 60, 120, 180, 240, 300)):
        rad = math.radians(angle)
        parts.append(_part(
            f"petal_{index}", "sphere", {"radius": 0.30}, color,
            "center", position="center",
            offset_x=0.46 * math.cos(rad),
            offset_y=0.46 * math.sin(rad),
        ))
    return parts

def _person_model(color):
    torso_h = 1.4
    return [
        _part("torso", "cube", {"dx": 0.55, "dy": 0.85, "dz": torso_h},
              color),
        _part("neck", "cylinder", {"radius": 0.10, "height": 0.15},
              "tan", "torso", position="top"),
        _part("head", "sphere", {"radius": 0.25}, "tan", "neck",
              position="top"),
        _part("hair", "sphere", {"radius": 0.26}, "brown", "head",
              position="top"),
        _part("arm_left", "cylinder", {"radius": 0.09, "height": 1.15},
              color, "torso", position="left", offset_z=-0.05),
        _part("arm_right", "cylinder", {"radius": 0.09, "height": 1.15},
              color, "torso", position="right", offset_z=-0.05),
        _part("hand_left", "sphere", {"radius": 0.11}, "tan", "arm_left",
              position="bottom"),
        _part("hand_right", "sphere", {"radius": 0.11}, "tan", "arm_right",
              position="bottom"),
        _part("leg_left", "cylinder", {"radius": 0.11, "height": 1.4},
              "blue", "torso", position="bottom", offset_x=-0.15),
        _part("leg_right", "cylinder", {"radius": 0.11, "height": 1.4},
              "blue", "torso", position="bottom", offset_x=0.15),
        _part("foot_left", "cube", {"dx": 0.18, "dy": 0.38, "dz": 0.10},
              "black", "leg_left", position="bottom", offset_y=0.10),
        _part("foot_right", "cube", {"dx": 0.18, "dy": 0.38, "dz": 0.10},
              "black", "leg_right", position="bottom", offset_y=0.10),
    ]

def _animal_model(kind, color):
    if kind == "bird":
        return [
            _part("body", "sphere", {"radius": 0.75}, color),
            _part("head", "sphere", {"radius": 0.45}, color, "body",
                  position="right", offset_z=0.30),
            _part("beak", "cone", {"radius": 0.13, "height": 0.35},
                  "orange", "head", position="right", rotation={"ry": 90}),
            _part("wing_left", "cube", {"dx": 0.50, "dy": 1.30, "dz": 0.08},
                  "darkgray", "body", position="front", rotation={"rz": 15}),
            _part("wing_right", "cube", {"dx": 0.50, "dy": 1.30, "dz": 0.08},
                  "darkgray", "body", position="back", rotation={"rz": -15}),
            _part("tail", "cube", {"dx": 0.95, "dy": 0.45, "dz": 0.07},
                  color, "body", position="left"),
            _part("leg_left", "cylinder", {"radius": 0.07, "height": 0.6},
                  "orange", "body", position="bottom", offset_x=-0.25),
            _part("leg_right", "cylinder", {"radius": 0.07, "height": 0.6},
                  "orange", "body", position="bottom", offset_x=0.25),
        ]
    if kind == "cat":
        return [
            _part("body", "cube", {"dx": 1.7, "dy": 0.72, "dz": 0.72},
                  color),
            _part("head", "sphere", {"radius": 0.45}, color, "body",
                  position="right", offset_z=0.20),
            _part("ear_left", "cone", {"radius": 0.16, "height": 0.32},
                  color, "head", position="top", offset_y=-0.22),
            _part("ear_right", "cone", {"radius": 0.16, "height": 0.32},
                  color, "head", position="top", offset_y=0.22),
            _part("nose", "sphere", {"radius": 0.08}, "pink", "head",
                  position="right"),
            _part("tail", "cylinder", {"radius": 0.07, "height": 1.1},
                  color, "body", position="left", rotation={"ry": -50}),
            _part("leg_fl", "cylinder", {"radius": 0.10, "height": 0.70},
                  color, "body", position="bottom",
                  offset_x=0.55, offset_y=-0.24),
            _part("leg_fr", "cylinder", {"radius": 0.10, "height": 0.70},
                  color, "body", position="bottom",
                  offset_x=0.55, offset_y=0.24),
            _part("leg_bl", "cylinder", {"radius": 0.10, "height": 0.70},
                  color, "body", position="bottom",
                  offset_x=-0.55, offset_y=-0.24),
            _part("leg_br", "cylinder", {"radius": 0.10, "height": 0.70},
                  color, "body", position="bottom",
                  offset_x=-0.55, offset_y=0.24),
        ]
    if kind == "dog":
        return [
            _part("body", "cube", {"dx": 2.0, "dy": 0.80, "dz": 0.85},
                  color),
            _part("head", "sphere", {"radius": 0.50}, color, "body",
                  position="right", offset_z=0.25),
            _part("snout", "cube", {"dx": 0.40, "dy": 0.38, "dz": 0.28},
                  color, "head", position="right"),
            _part("nose", "sphere", {"radius": 0.09}, "black", "snout",
                  position="right"),
            _part("ear_left", "cube", {"dx": 0.12, "dy": 0.22, "dz": 0.30},
                  color, "head", position="top", offset_y=-0.30),
            _part("ear_right", "cube", {"dx": 0.12, "dy": 0.22, "dz": 0.30},
                  color, "head", position="top", offset_y=0.30),
            _part("tail", "cylinder", {"radius": 0.08, "height": 0.85},
                  color, "body", position="left", rotation={"ry": -55}),
            _part("leg_fl", "cylinder", {"radius": 0.12, "height": 0.80},
                  color, "body", position="bottom",
                  offset_x=0.70, offset_y=-0.25),
            _part("leg_fr", "cylinder", {"radius": 0.12, "height": 0.80},
                  color, "body", position="bottom",
                  offset_x=0.70, offset_y=0.25),
            _part("leg_bl", "cylinder", {"radius": 0.12, "height": 0.80},
                  color, "body", position="bottom",
                  offset_x=-0.70, offset_y=-0.25),
            _part("leg_br", "cylinder", {"radius": 0.12, "height": 0.80},
                  color, "body", position="bottom",
                  offset_x=-0.70, offset_y=0.25),
        ]
    if kind in {"horse", "zebra", "cow", "giraffe"}:
        body_dz = 1.15 if kind == "giraffe" else 0.95
        parts = [
            _part("body", "cube", {"dx": 2.2, "dy": 1.0, "dz": body_dz},
                  color),
        ]
        if kind == "giraffe":
            parts += [
                _part("neck", "cylinder",
                      {"radius": 0.30, "height": 2.4}, color,
                      "body", position="right", offset_z=1.10),
                _part("head", "sphere", {"radius": 0.36}, color, "neck",
                      position="top"),
            ]
        else:
            parts += [
                _part("neck", "cube", {"dx": 0.40, "dy": 0.45, "dz": 0.65},
                      color, "body", position="right", offset_z=0.40),
                _part("head", "sphere", {"radius": 0.44}, color, "neck",
                      position="top"),
            ]
        if kind == "cow":
            parts += [
                _part("horn_left", "cone",
                      {"radius": 0.06, "height": 0.26}, "white",
                      "head", position="top", offset_y=-0.20),
                _part("horn_right", "cone",
                      {"radius": 0.06, "height": 0.26}, "white",
                      "head", position="top", offset_y=0.20),
            ]
        parts += [
            _part("tail", "cylinder", {"radius": 0.07, "height": 1.05},
                  color, "body", position="left", rotation={"ry": -45}),
            _part("leg_fl", "cylinder", {"radius": 0.13, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=0.75, offset_y=-0.30),
            _part("leg_fr", "cylinder", {"radius": 0.13, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=0.75, offset_y=0.30),
            _part("leg_bl", "cylinder", {"radius": 0.13, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=-0.75, offset_y=-0.30),
            _part("leg_br", "cylinder", {"radius": 0.13, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=-0.75, offset_y=0.30),
        ]
        return parts
    if kind == "sheep":
        return [
            _part("wool", "sphere", {"radius": 0.95}, "white"),
            _part("head", "sphere", {"radius": 0.36}, "black", "wool",
                  position="right"),
            _part("ear_left", "cube", {"dx": 0.09, "dy": 0.18, "dz": 0.09},
                  "black", "head", position="front"),
            _part("ear_right", "cube", {"dx": 0.09, "dy": 0.18, "dz": 0.09},
                  "black", "head", position="back"),
            _part("leg_fl", "cylinder", {"radius": 0.09, "height": 0.8},
                  "black", "wool", position="bottom",
                  offset_x=0.45, offset_y=-0.30),
            _part("leg_fr", "cylinder", {"radius": 0.09, "height": 0.8},
                  "black", "wool", position="bottom",
                  offset_x=0.45, offset_y=0.30),
            _part("leg_bl", "cylinder", {"radius": 0.09, "height": 0.8},
                  "black", "wool", position="bottom",
                  offset_x=-0.45, offset_y=-0.30),
            _part("leg_br", "cylinder", {"radius": 0.09, "height": 0.8},
                  "black", "wool", position="bottom",
                  offset_x=-0.45, offset_y=0.30),
        ]
    if kind == "elephant":
        return [
            _part("body", "cube", {"dx": 2.4, "dy": 1.4, "dz": 1.5},
                  color),
            _part("head", "sphere", {"radius": 0.70}, color, "body",
                  position="right", offset_z=0.20),
            _part("trunk", "cylinder", {"radius": 0.18, "height": 1.5},
                  color, "head", position="bottom", offset_x=0.30),
            _part("ear_left", "cube", {"dx": 0.65, "dy": 0.10, "dz": 0.85},
                  color, "head", position="front"),
            _part("ear_right", "cube", {"dx": 0.65, "dy": 0.10, "dz": 0.85},
                  color, "head", position="back"),
            _part("leg_fl", "cylinder", {"radius": 0.20, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=0.80, offset_y=-0.40),
            _part("leg_fr", "cylinder", {"radius": 0.20, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=0.80, offset_y=0.40),
            _part("leg_bl", "cylinder", {"radius": 0.20, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=-0.80, offset_y=-0.40),
            _part("leg_br", "cylinder", {"radius": 0.20, "height": 1.4},
                  color, "body", position="bottom",
                  offset_x=-0.80, offset_y=0.40),
        ]
    if kind == "bear":
        return [
            _part("body", "cube", {"dx": 2.0, "dy": 1.05, "dz": 1.15},
                  color),
            _part("head", "sphere", {"radius": 0.58}, color, "body",
                  position="right", offset_z=0.35),
            _part("ear_left", "sphere", {"radius": 0.16}, color, "head",
                  position="top", offset_y=-0.28),
            _part("ear_right", "sphere", {"radius": 0.16}, color, "head",
                  position="top", offset_y=0.28),
            _part("leg_fl", "cylinder", {"radius": 0.17, "height": 1.0},
                  color, "body", position="bottom",
                  offset_x=0.70, offset_y=-0.35),
            _part("leg_fr", "cylinder", {"radius": 0.17, "height": 1.0},
                  color, "body", position="bottom",
                  offset_x=0.70, offset_y=0.35),
            _part("leg_bl", "cylinder", {"radius": 0.17, "height": 1.0},
                  color, "body", position="bottom",
                  offset_x=-0.70, offset_y=-0.35),
            _part("leg_br", "cylinder", {"radius": 0.17, "height": 1.0},
                  color, "body", position="bottom",
                  offset_x=-0.70, offset_y=0.35),
        ]
    return [
        _part("body", "sphere", {"radius": 1.05}, color),
        _part("head", "sphere", {"radius": 0.72}, color, "body",
              position="top"),
        _part("ear_left", "sphere", {"radius": 0.22}, color, "head",
              position="top", offset_x=-0.48),
        _part("ear_right", "sphere", {"radius": 0.22}, color, "head",
              position="top", offset_x=0.48),
        _part("muzzle", "sphere", {"radius": 0.26}, "tan", "head",
              position="front"),
        _part("arm_left", "sphere", {"radius": 0.32}, color, "body",
              position="left"),
        _part("arm_right", "sphere", {"radius": 0.32}, color, "body",
              position="right"),
        _part("leg_left", "sphere", {"radius": 0.42}, color, "body",
              position="bottom", offset_x=-0.55),
        _part("leg_right", "sphere", {"radius": 0.42}, color, "body",
              position="bottom", offset_x=0.55),
    ]

# Dispatcher
def _catalog_model(kind, color):
    # Basic primitives — single-shape objects by design.
    if kind == "cube":
        return [_part("body", "cube", {"dx": 2.0, "dy": 2.0, "dz": 2.0}, color)]
    if kind == "box":
        return [_part("body", "cube", {"dx": 3.2, "dy": 2.0, "dz": 1.8}, color)]
    if kind == "sphere":
        return [_part("body", "sphere", {"radius": 1.2}, color)]
    if kind == "cylinder":
        return [_part("body", "cylinder", {"radius": 0.9, "height": 2.4}, color)]
    if kind == "cone":
        return [_part("body", "cone", {"radius": 1.1, "height": 2.4}, color)]
    if kind == "pyramid":
        return [_part("body", "pyramid", {"base": 2.2, "height": 2.4}, color)]

    # Vehicles
    if kind == "car":
        return _car_model(color)
    if kind == "truck":
        return _truck_model(color)
    if kind == "bus":
        return _bus_model(color)
    if kind == "motorcycle":
        return _motorcycle_model(color)
    if kind == "bicycle":
        return _bicycle_model(color)
    if kind == "train":
        return _train_model(color)
    if kind == "plane":
        return _plane_model(color)
    if kind == "helicopter":
        return _helicopter_model(color)
    if kind == "rocket":
        return _rocket_model(color)
    if kind == "boat":
        return _boat_model(color)

    # Furniture
    if kind in {"table", "dining_table"}:
        return _table_model(color)
    if kind == "bench":
        return _bench_model(color)
    if kind == "chair":
        return _chair_model(color)
    if kind == "sofa":
        return _sofa_model(color)
    if kind == "bed":
        return _bed_model(color)
    if kind == "bookshelf":
        return _bookshelf_model(color)
    if kind == "lamp":
        return _lamp_model(color)

    # Devices and appliances
    if kind in {"mouse", "cell_phone", "laptop", "keyboard", "remote"}:
        return _device_model(kind, color)
    if kind in {"clock", "tv", "microwave", "oven", "toaster",
                "refrigerator", "toilet", "sink"}:
        return _appliance_model(kind, color)

    # Containers and utensils
    if kind in {"bottle", "vase", "book", "suitcase", "backpack",
                "cup", "bowl"}:
        return _container_model(kind, color)
    if kind in {"knife", "spoon", "fork", "scissors", "toothbrush"}:
        return _utensil_model(kind, color)
    if kind in {"pencil", "eraser", "ruler"}:
        return _writing_model(kind, color)

    # Food
    if kind in {"apple", "orange", "banana", "donut", "pizza", "cake",
                "sandwich", "hot_dog", "broccoli", "carrot"}:
        return _food_model(kind, color)

    # Sports
    if kind in {"sports_ball", "frisbee", "skateboard", "snowboard",
                "surfboard", "skis", "baseball_bat", "tennis_racket",
                "baseball_glove", "kite"}:
        return _sports_model(kind, color)

    # Accessories
    if kind in {"traffic_light", "fire_hydrant", "stop_sign",
                "parking_meter", "umbrella", "tie", "handbag",
                "potted_plant", "hair_drier"}:
        return _accessory_model(kind, color)

    # House
    if kind == "house":
        return [
            _part("walls", "cube", {"dx": 3.8, "dy": 3.0, "dz": 2.6}, color),
            _part("roof", "pyramid", {"base": 4.4, "height": 1.6}, "red",
                  "walls", position="top"),
            _part("door", "cube", {"dx": 0.85, "dy": 0.10, "dz": 1.45},
                  "brown", "walls", position="front", offset_z=-0.55),
            _part("window_left", "cube", {"dx": 0.65, "dy": 0.10, "dz": 0.65},
                  "lightblue", "walls", position="front",
                  offset_x=-1.15, offset_z=0.55),
            _part("window_right", "cube",
                  {"dx": 0.65, "dy": 0.10, "dz": 0.65}, "lightblue",
                  "walls", position="front",
                  offset_x=1.15, offset_z=0.55),
            _part("chimney", "cube", {"dx": 0.38, "dy": 0.38, "dz": 1.0},
                  "brown", "walls", position="top", offset_x=1.15),
        ]

    # Nature, people, animals
    if kind in {"tree", "flower"}:
        return _nature_model(kind, color)
    if kind == "person":
        return _person_model(color)
    if kind in {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "teddy_bear"}:
        return _animal_model(kind, color)

    # Generic fallback for unrecognised kinds
    return [_part("body", "cube", {"dx": 1.8, "dy": 1.4, "dz": 1.2}, color)]

def create_preset(description):
    # Return a fast model for a known object, or None for AI generation.
    text = re.sub(r"[^a-z0-9\s]", " ", description.lower())
    text = re.sub(r"\s+", " ", text).strip()
    color = next(
        (value for name, value in COLORS.items()
         if re.search(rf"\b{name}\b", text)),
        "gray",
    )
    matches = []
    for kind, aliases in ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                matches.append((len(alias), kind))
    if matches:
        _, kind = max(matches)
        return {
            "primitives": deepcopy(_catalog_model(kind, color)),
            "source": "catalog",
        }
    return None