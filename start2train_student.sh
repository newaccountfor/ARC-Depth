# training with pose-adaptive reconstruction loss
# python train.py --pose_idea --scheduler_step_size 14  --model_name only_pose --png --data_path ~/kitti

# training with region-sensitive reconstruction loss
# python train.py --reconstruction_idea --scheduler_step_size 14 --model_name only_reconstruction --png --data_path ~/kitti

# training with both pose-adaptive reconstruction loss and region-sensitive reconstruction loss
# python train.py --reconstruction_idea --pose_idea --scheduler_step_size 14 --model_name pose_and_reconstruction --png --data_path ~/kitti

# training with pose-adaptive reconstruction loss, region-sensitive reconstruction loss and bidirectional distillation loss
# need to provide a path of student net to predict depth map as input of teacher net, together with the path of tacher net and
python train.py --reconstruction_idea --pose_idea --use_teacher --teacher_model_path ~/model/teacher_pose/models/weights_X --student_model_input_of_disp_for_t ~/model/pose_and_reconstruction/models/weights_X --scheduler_step_size 14 --model_name student_teacher_pose_reconstruction_from_scratch --png --data_path ~/kitti

# when continue trainging with pretrained weights, use this
# python train.py --reconstruction_idea --pose_idea --use_teacher --teacher_model_path ~/model/teacher_pose/models/weights_X  --student_model_input_of_disp_for_t ~/model/pose_and_reconstruction/models/weights_X --scheduler_step_size 14  --model_name student_teacher_pose_reconstruction --load_weights_folder ~/pretrained-model/models/weights_X --png --data_path ~/kitti