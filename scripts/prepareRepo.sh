# ------------------------- BODY, FACE AND HAND MODELS -------------------------
# Downloading body pose (COCO and MPI), face and hand models
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"
POSE_FOLDER="pose/"
FACE_FOLDER="face/"
HAND_FOLDER="hand/"

# ------------------------- POSE MODELS -------------------------
# Body (COCO)
# COCO_FOLDER=${POSE_FOLDER}"coco/"
# COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"

cd ../
mkdir Videos
mkdir Data
mkdir Models
mkdir Others

cd Models
# wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${COCO_FOLDER}

