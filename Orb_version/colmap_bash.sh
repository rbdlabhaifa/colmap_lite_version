#!/bin/bash

# use ORB
use_orb_version=true
USER_NAME="rbdstudent"

# Get path to image and database folders:
while getopts p: flag
do
    case "${flag}" in
        p) DB_PATH=${OPTARG};;
    esac
done

start_program_time="$(date -u +%s)"

# Start room scan:
#python3 "/home/$USER_NAME/colmap/Orb_version/scan_script.py" \
#  --output "$DB_PATH"

# Workplace preparation (images and camera pose)
start="$(date -u +%s)"

if [ "$use_orb_version" = true ] ; then
  python3 "/home/$USER_NAME/colmap/Orb_version/workplace_preparation.py"  \
  	  --workspace_path "$DB_PATH"
else
  mkdir $DB_PATH/images
  ffmpeg -i "$DB_PATH/outpy.h264" -r 0.5 -f image2 "$DB_PATH/images/image%d.jpg"
fi

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "Workplace preparation (images and camera pose)  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

if [ "$use_orb_version" = true ] ; then
  # Extract and match ORB features
  python3 "/home/$USER_NAME/colmap/Orb_version/detect_and_compute_keypoints.py" \
     --path "$DB_PATH/images/" \
     --db_path "$DB_PATH"

  start="$(date -u +%s)"

  colmap sequential_matcher \
    --database_path "$DB_PATH/database.db"

  end="$(date -u +%s)"
  elapsed="$(($end-$start))"
  echo "Two View Geometry  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

  start="$(date -u +%s)"

  colmap point_triangulator \
     --database_path $DB_PATH/database.db \
     --image_path $DB_PATH/images \
     --input_path $DB_PATH/camera_poses \
     --output_path $DB_PATH

  end="$(date -u +%s)"
  elapsed="$(($end-$start))"
  echo "Run sparse reconstruction  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

else
  colmap automatic_reconstructor \
  	  --workspace_path "$DB_PATH" \
	    --image_path "$DB_PATH/images/" \
	    --data_type="video" \
	    --quality="extreme"

	cp -i "$DB_PATH/sparse/0/images.txt" "$DB_PATH"
  cp -i "$DB_PATH/sparse/0/points3D.txt" "$DB_PATH"
fi

# Run the conversion to RBD data format:
start="$(date -u +%s)"

python3 "/home/$USER_NAME/colmap/Orb_version/db_conversion.py" \
    --input_path  "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "output format prep [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Run exit room algorithm:
start="$(date -u +%s)"
python3 "/home/$USER_NAME/colmap/Orb_version/Find_exit/find_room_exit.py" \
    --path "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "exit room algorithm [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

elapsed="$(($end-$start_program_time))"
echo "total time [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"