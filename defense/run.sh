#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_data
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_data/result.csv
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_FILE - the classification result for each image
#

INPUT_DIR=$1
OUTPUT_FILE=$2

python env_test.py \
  "${INPUT_DIR}" \
  "${OUTPUT_FILE}" \
