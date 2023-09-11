# for pose-adaptive reconstruction loss only
# python evaluate_depth.py   --load_weights_folder ~/model/only_pose/'time'/models/weights_X --eval_mono --data_path ~/kitti

# for region-sensitive reconstruction loss
# python evaluate_depth.py   --load_weights_folder ~/model/only_reconstruction/'time'/models/weights_X --eval_mono --data_path ~/kitti

# for both pose-adaptive reconstruction loss and region-sensitive reconstruction loss
# python evaluate_depth.py   --load_weights_folder ~/model/pose_and_reconstruction/'time'/models/weights_X --eval_mono --data_path ~/kitti

# for pose-adaptive reconstruction loss, region-sensitive reconstruction loss and bidirectional distillation loss
python evaluate_depth.py   --load_weights_folder ~/model/student_teacher_pose_reconstruction/'time'/models/weights_X --eval_mono --data_path ~/kitti