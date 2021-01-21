#!/bin/bash
source /home/fares/venv37/bin/activate

# Get path to image and database folders:
while getopts p: flag
do
    case "${flag}" in
        p) DB_PATH=${OPTARG};;
    esac
done

start_program_time="$(date -u +%s)"

# Start room scan:
echo "Start room scan"

python3 /home/fares/colmap_orb/scan_script.py\
   --output "$DB_PATH"

mkdir $DB_PATH/images
ffmpeg -i "$DB_PATH/outpy.avi" -r 0.8 -f image2 "$DB_PATH/images/image%d.jpg"


/home/fares/colmap/build/src/exe/colmap feature_extractor \
   --database_path "$DB_PATH/database.db" \
   --image_path "$DB_PATH/images/" \
   --ImageReader.single_camera=1 \
   --SiftExtraction.max_num_features=3000

# Extract and match ORB features, and save them in a database.db file:
echo "Extract and Match ORB features"
python3 /home/fares/colmap_orb/detect_and_compute_keypoints.py\
    --path "$DB_PATH/images/" \
    --db_path "$DB_PATH"

# Run feature matcher (Geometric Verification):
echo "Geometric Verification"
start="$(date -u +%s)"

/home/fares/colmap/build/src/exe/colmap sequential_matcher \
    --database_path "$DB_PATH/database.db"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "bad match removal  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Run sparse reconstruction:
echo "Run sparse reconstruction"
start="$(date -u +%s)"

/home/fares/colmap/build/src/exe/colmap mapper \
    --database_path "$DB_PATH/database.db" \
    --image_path "$DB_PATH/images/" \
    --output_path "$DB_PATH" \
    --Mapper.extract_colors=0 \
    --Mapper.ba_global_use_pba=1

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "bundle adjustment  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Merge in case of multiple models:
if [ -d "$DB_PATH/1/" ]; then
    echo "Merging multiple models"
    start="$(date -u +%s)"

    /home/fares/colmap/build/src/exe/colmap model_merger \
        --input_path1 "$DB_PATH/0/" \
        --input_path2 "$DB_PATH/1/" \
        --output_path "$DB_PATH/0/"
    if [ -d "$DB_PATH/2/" ]; then
        /home/fares/colmap/build/src/exe/colmap model_merger \
            --input_path1 "$DB_PATH/0/" \
            --input_path2 "$DB_PATH/2/" \
            --output_path "$DB_PATH/0/"
        if [ -d "$DB_PATH/3/" ]; then
            /home/fares/colmap/build/src/exe/colmap model_merger \
                --input_path1 "$DB_PATH/0/" \
                --input_path2 "$DB_PATH/3/" \
                --output_path "$DB_PATH/0/"
        fi
    fi

    # Run another global bundle adjustment after the merge:
    /home/fares/colmap/build/src/exe/colmap bundle_adjuster \
        --input_path "$DB_PATH/0/" \
        --output_path "$DB_PATH/0/"

    end="$(date -u +%s)"
    elapsed="$(($end-$start))"
    echo "merge multi-models [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"
fi

# Place model with maximal number of 3D points as model-0:
python3 /home/fares/colmap_orb/find_biggest_model.py \
    --input_path  "$DB_PATH"

# Run the conversion to RBD data format:
echo "Run conversion to RBD data format"
start="$(date -u +%s)"

# Convert from binary to txt output format:
/home/fares/colmap/build/src/exe/colmap model_converter \
    --input_path "$DB_PATH/0/" \
    --output_path "$DB_PATH" \
    --output_type TXT

python3 /home/fares/colmap_orb/db_conversion.py \
    --input_path  "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "output format prep [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Run exit room algorithm:
echo "Run exit room algorithm"
python3 /home/fares/colmap_orb/Find_exit/find_room_exit.py \
    --path "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "exit room algorithm [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

elapsed="$(($end-$start_program_time))"
echo "total time [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"


