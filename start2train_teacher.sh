# training of the teacher network
# need to provide a path of student net to predict depth map as input of teacher net, together with the path of tacher net and
python train_teacher.py --pose_idea --scheduler_step_size 14 --model_name teacher_pose --png --data_path ~/kitti --student_model_input_of_disp_for_t ~/model/pose_and_reconstruction/models/weights_X