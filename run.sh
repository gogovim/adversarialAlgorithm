#!/bin/bash
#
# run.sh is the entry point of the submission.
# nvidia-docker run -v ${INPUT_DIR}:/input_images -v ${OUTPUT_DIR}:/output_images
#       -w /competition ${DOCKER_IMAGE_NAME} sh ./run.sh /input_images /output_images
# where:
#   INPUT_DIR - directory with input png images
#   OUTPUT_DIR - directory with output png images
#
inputs=(
	"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train" \
	"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test")
models=("vgg16" "resnet101" "densenet" \
	"InceptionResNetV2" "InceptionV4" \
	"Inception3")
pre_trains=("HGD_299_model.dat" "Inception3_clean_299_model.dat" \
	"InceptionResNetV2_clean_299_model.dat" "Inceptionv4_clean_299_model.dat" \
	"resnet101_adversial_model.dat" "resnet101_clean_192_model.dat" \
	"resnet101_clean_299_model.dat" "resnet101_model.dat" \
	"vgg16_clean_299_model.dat" "vgg16_model.dat" \
	"densenet_clean_166_model.dat")

#valid
#python train.py --mode="valid" --input_dir="${inputs[ch0]}" --model_name="${models[ch1]}" --pho_size=299 --save_dir=./output \
#--pre_train="./weights/${pre_trains[ch2]}" --batch_size=8

#test
#ch0=0
#ch1=1
#ch2=5
#python train.py --mode="test" --test_dir="${inputs[ch0]}" --adversarial_test_dir="${inputs[0]}" --model_name="${models[ch1]}" --pho_size=299 --save_dir=./output \
#--pre_train="./weights/${pre_trains[ch2]}" --batch_size=8

#generate adversarial examples
#ch0=0
#ch1=1
#ch2=6
#python train.py --mode="generate" --input_dir="${inputs[ch0]}" --model_name="${models[ch1]}" --pho_size=299 --save_dir=/home/data/wanghao/tianchi/data \
#--pre_train="./weights/${pre_trains[ch2]}" --batch_size=8 --adversarial_method="i-fgsm" --eps=0.03 --iteration=10

#test
#ch0=5
#ch1=1
#ch2=6
#python train.py --mode="test" --test_dir="${inputs[4]}" --adversarial_test_dir="${inputs[ch0]}" --model_name="${models[ch1]}" --pho_size=299 --save_dir=./output \
#--pre_train="./weights/${pre_trains[ch2]}" --batch_size=8

#clean train
python train.py --mode="train" --train_dir="${inputs[0]}" --valid_dir="${inputs[1]}" --model_name="${models[2]}" --pho_size=299 --save_dir=./weights --batch_size=16 --pre_train=./weights/densenet_clean_299_model.dat --lr=0.00001

#adversarial train


#python main.py --input_dir="${INPUT_DIR}" --output_dir="${OUTPUT_DIR}"
#echo $INPUT_DIR
#echo $OUTPUT_DIR