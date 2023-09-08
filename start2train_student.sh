
#python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed

# only pose idea
# CUDA_VISIBLE_DEVICES=1 python train.py --pose_idea --scheduler_step_size 14  --batch 10  --model_name only_pose --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/diffnet_640x192

# only reconstruction idea
#CUDA_VISIBLE_DEVICES=1 python train.py --reconstruction_idea --scheduler_step_size 14  --batch 12  --model_name only_reconstruction --png --data_path /home/sdb1/ouyuxiang/kitti/kitti #--load_weights_folder /home/sdb1/ouyuxiang/pretrained-model/diffnet_640x192
# python train.py --reconstruction_idea --scheduler_step_size 14  --batch 8  --model_name only_reconstruction_get_f_first --png --data_path /home/sdb1/ouyuxiang/kitti/kitti --get_f_first
# pose idea and reconstruction idea
#CUDA_VISIBLE_DEVICES=1 python train.py --reconstruction_idea --pose_idea --scheduler_step_size 14  --batch 10  --model_name pose_and_reconstruction --png --data_path ../datasets/ --load_weights_folder /home/inspur/MAX_SPACE/yangli/pretrained-model/diffnet_640x192 --get_f_first

# pose_idea and reconstruction idea and teacher and get_f_first
# python train.py --reconstruction_idea --pose_idea --use_teacher --teacher_model_path /home/inspur/MAX_SPACE/yangli/model/teacher_pose/17032023-10\:52\:55/models/weights_13 --student_model_input_of_disp_for_t /home/inspur/MAX_SPACE/yangli/pretrained-model/weights_5 --scheduler_step_size 14  --batch 12  --model_name student_teacher_pose_reconstruction_from_scratch --png --data_path ../datasets/
CUDA_VISIBLE_DEVICES=0 python train.py --reconstruction_idea --pose_idea --use_teacher --teacher_model_path /home/sdb1/ouyuxiang/biaobiaobiao/model/teacher_pose/models/weights_13 --student_model_input_of_disp_for_t /home/sdb1/ouyuxiang/pretrained-model/weights_5 --scheduler_step_size 14  --batch 4 --model_name student_teacher_pose_reconstruction_from_scratch --png --data_path /home/sdb1/ouyuxiang/kitti/kitti --num_epochs 24