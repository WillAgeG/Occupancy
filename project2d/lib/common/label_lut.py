LABEL_LUT_OLD = {"Unknown": 0,
    "unlabeled": 0,
    "unlabelled": 0,
    "curb": 1,
    "fence": 1,
    "moving_now": 3,
    "occupied": 1,
    "pole": 1,
    "road": 2,
}

# extended part, project any dataset to minimal
LABEL2INT_extended = {
    "Unknown": 0,
    "unlabeled": 0,
    "unlabelled": 0,
    "curb": 2,
    "fence": 2,
    "moving_now": 3,
    "occupied": 2,
    "pole": 2,
    "road": 1,
    "vehicle": 3,
    "bus": 3,
    "truck": 3,
    "motorbike": 3,
    "human": 3,
    "bicycle": 3,
    "trfc": 3,
    "construction_vehicle": 3,
    "unknown": 0,
    "barrier": 2,
    "van": 3,
    "scooter": 3,
    "trailer": 3,
    "0": 0
}

# annotations part, has labels only from annotations
label2color_annotations = {
    "curb": [255, 128, 0],
    "fence": [0, 255, 255],
    "moving_now": [245, 150, 100],
    "occupied": [255, 255, 50],
    "pole": [255, 255, 0],
    "road": [255, 0, 255],
    "unlabeled": [0, 0, 0],
    "unlabelled": [0, 0, 0],
}

# minimal part, only labels which we are interested in
int2label_minimal = {0: "unlabeled",
    1: "road",
    2: "occupied",
    3: "moving_now",
}

label2int_minimal = {v: k for k, v in int2label_minimal.items()}
int2color_minimal = {k: label2color_annotations[v] for k, v in int2label_minimal.items()}

# TODO(msmelkumov): fix data collection to label maps straight
# mapping from static images to classes
static_lut = {
    255: 0,  # unlabeled
    120: 1,  # road
    0: 2,  # occupied
}


CLASS_MAPPING_tracker_to_occupancy = {
    "Pedestrian": "moving_now",
    "Car": "moving_now",
    "Bicycle": "moving_now",
    "Bus": "moving_now",
    "Truck": "moving_now",
    "Train": "moving_now",
    "Tram": "moving_now",
    "Motorcycle": "moving_now",
    "KickScooter": "moving_now",
    "CarTrailer": "moving_now",
    "TruckTrailer": "moving_now",
    "ConstructionVehicle": "moving_now",
    "SingleTruck": "moving_now",
    "Golfcar": "moving_now",
    "Wheelchair": "moving_now",
    "BabyCarriage": "moving_now",
    "Animal": "moving_now",
    "Cone": "occupied",
    "TrafficBarrier": "occupied",
    "TrafficBeacon": "occupied",
    "Pillar": "occupied",
    "Tollbooth": "occupied",
    "RoadBuffer": "occupied",
    "TrafficSignStand": "occupied",
    "TollboothPillar": "occupied",
    "SignalTrailer": "occupied",
    "BoomBarrier": "occupied",
    "TrafficSign": "occupied",
}

IGNORE_CLASSES_tracker_to_occupancy = set(["Dust", "Water", "FakeCar", "Unknown"])
