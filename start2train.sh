
#python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed
#CUDA_VISIBLE_DEVICES=1 python train.py --scheduler_step_size 14  --batch 12  --model_name lr-4-v1 --png --data_path ../../KITTI/datasets/raw_data --load_weights_folder 
python train.py --scheduler_step_size 14  --batch 12  --model_name load_pose_and_depth_4pose_again --png --data_path ../../KITTI/datasets/raw_data --load_weights_folder /home/sba/wubiao/newidea-newbackbone/pretrained-model/diffnet_640x192
