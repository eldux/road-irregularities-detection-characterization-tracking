This is repository with YOLOv5 based code for detection, characterization, tracking of road surface irregularities such as potholes, manholes and speed reducing bumps/tables including rised pedestrian crossings and rised intersections. 

You need to install newest ultralytics packages or ones in the requirements.txt or create conda environment with required packages to run it.

YOu can run full script from yolov5 folder. The run command should look like that:
python detect_trio_svo_pc6_sbs.py --view-img --weights /path_to_closed_repository/yolov5/runs/detection/train/weights/best.pt --conf-thres 0.6 --source /path_to_svo2_file/svo2_file_name.svo2 --half --tracking-overlay --save-csv

This will output right camera images and bird-eye-view images side-by-side with all detections and characterization of the object. After run, CSV file with all characterization will be generated, as well as MP4 video file with recorded screen view. You can adjust the conf-thres to get more or less sensitive detections, you can turn off preview by removing view-img option. 

The base code was provided with GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007, there fore it is published with same LICENSE type and should retain information about the original author and the author of this modified version if you use it. The several files in the original code were modified to support SVO2 files, to include ZED SDK, OPEN3D processing, and modified viewer, video and CSV outputs.

THIS IS EXPERIMENTAL CODE. AUTHOR DO NOT TAKE ANY RESPONSIBILITY FOR ANYONE USING IT FOR ANY WORK OR FIELD OF APPLICATION.
