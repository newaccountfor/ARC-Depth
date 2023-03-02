
#python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed
CUDA_VISIBLE_DEVICES=1 python train.py --train_which 1 --scheduler_step_size 14  --batch 12  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/diffnet_640x192
