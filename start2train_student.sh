
#python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed

# only pose idea
CUDA_VISIBLE_DEVICES=1 python train.py --pose_idea --scheduler_step_size 14  --batch 12  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/weights_5

# only reconstruction idea
#CUDA_VISIBLE_DEVICES=1 python train.py --reconstruction_idea --scheduler_step_size 14  --batch 12  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/weights_5

# pose idea and reconstruction idea
#CUDA_VISIBLE_DEVICES=1 python train.py --reconstruction_idea --pose_idea --scheduler_step_size 14  --batch 12  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/weights_5

# pose idea and reconstruction idea and teacher
#CUDA_VISIBLE_DEVICES=1 python train.py --reconstruction_idea --pose_idea --use_teacher --teacher_model_path path_to_teacher_model student_model_input_of_disp_for_t path --scheduler_step_size 14  --batch 12  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/weights_5
