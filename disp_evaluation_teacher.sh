# for teacher net
# need a student net weights and a teacher net weights
python evaluate_depth_teacher.py   --load_weights_folder ~/model/pose_and_reconstruction/'time'/models/weights_X --eval_mono --data_path ~/kitti --load_weights_folder_teacher ~/teacher_pose/'time'/models/weights_X 

