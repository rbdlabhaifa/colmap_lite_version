#!/bin/bash

# Get path to image and database folders:
while getopts p: flag
do
    case "${flag}" in
        p) DB_PATH=${OPTARG};;
    esac
done
cd /home/rbdstudent/colmap/build/src/exe/
# Extract and match ORB features, and save them in a database.db file:
echo "Extract and Match"
python3 /home/rbdstudent/colmap/detect_and_compute_keypoints.py\
    --path "$DB_PATH" \
    --db_path "$DB_PATH"


# Run feature matcher (only outlier removal):
echo "Run feature matcher (only outlier removal)"
start="$(date -u +%s)"

./colmap  exhaustive_matcher \
    --database_path "$DB_PATH/database.db"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "bad match removal  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Run sparse reconstruction:
echo "Run sparse reconstruction"
start="$(date -u +%s)"

./colmap mapper \
    --database_path "$DB_PATH/database.db" \
    --image_path "$DB_PATH" \
    --output_path "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "bundle adjustment  [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Merge in case of multiple models:
if [ -d "$DB_PATH/1/" ]; then
    echo "Merging multiple models"
    start="$(date -u +%s)"

    ./colmap model_merger \
        --input_path1 "$DB_PATH/0/" \
        --input_path2 "$DB_PATH/1/" \
        --output_path "$DB_PATH/0/"
    if [ -d "$DB_PATH/2/" ]; then
        ./colmap model_merger \
            --input_path1 "$DB_PATH/0/" \
            --input_path2 "$DB_PATH/2/" \
            --output_path "$DB_PATH/0/"
        if [ -d "$DB_PATH/3/" ]; then
            ./colmap model_merger \
                --input_path1 "$DB_PATH/0/" \
                --input_path2 "$DB_PATH/3/" \
                --output_path "$DB_PATH/0/"
        fi
    fi
	
    # Run another global bundle adjustment after the merge:
    ./colmap bundle_adjuster \
        --input_path "$DB_PATH/0/" \
        --output_path "$DB_PATH/0/"

    end="$(date -u +%s)"
    elapsed="$(($end-$start))"
    echo "merge multi-models [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"
fi

# Place model with maximal number of 3D points as model-0:
python3 /home/rbdstudent/colmap/find_biggest_model.py \
    --input_path  "$DB_PATH"

# Run the conversion to RBD data format
echo "Run conversion to RBD data format"
start="$(date -u +%s)"

# Convert from binary to txt output format:
./colmap model_converter \
    --input_path "$DB_PATH/0/" \
    --output_path "$DB_PATH" \
    --output_type TXT

python3 /home/rbdstudent/colmap/db_conversion.py \
    --input_path  "$DB_PATH"

end="$(date -u +%s)"
elapsed="$(($end-$start))"
echo "output format prep [sec]: $elapsed" >> "$DB_PATH/sfm_log.txt"

# Clean-up - Remove unnecessary files:
rm -r "$DB_PATH/0/"
