import sys
sys.path.append('../src')
from processing import processKeypointsData

if __name__ == "__main__":
    processKeypointsData("Female_Fast", "Female_Fast", process_params="BIK", pose_model="SR")
    sys.exit(-1)
