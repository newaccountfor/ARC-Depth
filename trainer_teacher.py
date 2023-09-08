from __future__ import absolute_import, division, print_function
from datetime import datetime
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import torch.nn.functional as F

from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks

class Trainer:
    def __init__(self, options):
        now = datetime.now()
        current_time_date = now.strftime("%d%m%Y-%H:%M:%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name, current_time_date) 

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.opt.pose_idea == True or self.opt.reconstruction_idea == True, "must set one of pose idea and reconstruction_idea"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        #able if not using use_stereo or frame_ids !=0
        #use_stereo defualt disable
        #frame_ids defualt =[0,-1,1]

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.models["encoder_t"] = networks.ResnetEncoder(
            50, self.opt.weights_init == "pretrained", num_input_images=4)
        self.models["encoder_t"].to(self.device)
        self.parameters_to_train += list(self.models["encoder_t"].parameters())

        self.models["depth_t"] = networks.TeacherDecoder(
            self.models["encoder_t"].num_ch_enc, 1)
        self.models["depth_t"].to(self.device)
        self.parameters_to_train += list(self.models["depth_t"].parameters())

        para_sum = sum(p.numel() for p in self.models['encoder_t'].parameters())
        print('params in encoder_teacher',para_sum)

        para_sum = sum(p.numel() for p in self.models['depth_t'].parameters())
        print('params in depth decdoer',para_sum)


        # student for help teacher training

        self.student_help_teacher_encoder = networks.test_hr_encoder.hrnet18(True)
        self.student_help_teacher_encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
        self.student_help_teacher_decoder = networks.HRDepthDecoder(self.student_help_teacher_encoder.num_ch_enc, self.opt.scales)
        
        for model_type in ["encoder.pth", "depth.pth"]:
            model_path = os.path.join(self.opt.student_model_input_of_disp_for_t, model_type)
            pretrained_dict_for_student = torch.load(model_path)
            
            if model_type == "encoder.pth":
                # pretrained_dict_for_student = {k: v for k, v in pretrained_dict_for_student.items() if k in self.student_help_teacher_encoder}
                enc_model_dict = self.student_help_teacher_encoder.state_dict()
                self.student_help_teacher_encoder.load_state_dict({k: v for k, v in pretrained_dict_for_student.items() if k in enc_model_dict})
                for param in self.student_help_teacher_encoder.parameters():
                    param.requires_grad = False
                self.student_help_teacher_encoder.to(self.device)
            else:
                # pretrained_dict_for_student = {k: v for k, v in pretrained_dict_for_student.items() if k in self.student_help_teacher_decoder}
                dec_model_dict = self.student_help_teacher_decoder.state_dict()
                self.student_help_teacher_decoder.load_state_dict({k: v for k, v in pretrained_dict_for_student.items() if k in dec_model_dict})
                for param in self.student_help_teacher_decoder.parameters():
                    param.requires_grad = False
                self.student_help_teacher_decoder.to(self.device)

        if self.use_pose_net:  #use_pose_net = True
            if self.opt.pose_model_type == "separate_resnet":  #defualt=separate_resnet  choice = ['normal or shared']
                
                # Pose estimation's Encoder
                self.models["pose_encoder_t"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)#num_input_images=2
                
                # if we use pose idea, we use other two pose decoder
                if self.opt.pose_idea == True:
                    # Only output translation
                    self.models["pose_for_t"] = networks.PoseDecoder_for_t(
                        self.models["pose_encoder_t"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)

                    # only output rotation
                    self.models["pose_for_r"] = networks.PoseDecoder_for_r(
                        self.models["pose_encoder_t"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2)

                # Whether there is pose idea or not, pose for estimating R and T is required
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder_t"].num_ch_enc,
                        num_input_features=1,
                        num_frames_to_predict_for=2
                )

            if self.opt.pose_idea == True:
                self.models["pose_for_t"].cuda()
                self.models["pose_for_r"].cuda()
                self.parameters_to_train += list(self.models["pose_for_t"].parameters())
                self.parameters_to_train += list(self.models["pose_for_r"].parameters())

            self.models["pose_encoder_t"].cuda()
            self.models["pose"].cuda()
            self.parameters_to_train += list(self.models["pose_encoder_t"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate) #learning_rate=1e-4
        #self.model_optimizer = optim.Adam(self.parameters_to_train, 0.5 * self.opt.learning_rate)#learning_rate=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # feature generated
        self.backproject_feature = Backproject(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))
        self.project_feature = Project(self.opt.batch_size, int(self.opt.height/2), int(self.opt.width/2))
        self.backproject_feature.to(self.device)
        self.project_feature.to(self.device)

        
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset
                         }
        self.dataset_k = datasets_dict[self.opt.dataset]
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        #change trainset
        train_filenames_k = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames_k)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        #dataloader for kitti
        train_dataset_k = self.dataset_k(
            self.opt.data_path, train_filenames_k, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext = '.jpg')
        self.train_loader_k = DataLoader(
            train_dataset_k, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        #val_dataset = self.dataset(
        val_dataset = self.dataset_k( 
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)


        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch_k = train_dataset_k.__len__() // self.opt.batch_size

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset_k), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.init_time = time.time()
        self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Threads: " + str(torch.get_num_threads()))
        print("Training")
        self.set_train()
        self.every_epoch_start_time = time.time()
        
        for batch_idx, inputs in enumerate(self.train_loader_k):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000#log_fre 's defualt = 250
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                #self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        
        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))
   
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we first generate 3 disp images
            source1 = inputs[("color_aug", -1, 0)]   # -1 frame
            source0 = inputs[("color_aug", 0, 0)]    # 0 frame
            source2 = inputs[("color_aug", 1, 0)]    # 1 frame

            # disp1_help_teacher is a dict
            disp1_help_teacher = self.student_help_teacher_decoder(self.student_help_teacher_encoder(source1))
            disp0_help_teacher = self.student_help_teacher_decoder(self.student_help_teacher_encoder(source0))
            disp2_help_teacher = self.student_help_teacher_decoder(self.student_help_teacher_encoder(source2))

            # then fed then to teacher network
            teacher_input = torch.cat((source1, disp1_help_teacher[("disp", 0)], source0, disp0_help_teacher[("disp", 0)], source2, disp2_help_teacher[("disp", 0)]), 1)

            features = self.models["encoder_t"](teacher_input)
            outputs = self.models["depth_t"](features)

            # add student result to output dict to make selective supervised
            outputs.update(disp0_help_teacher)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
            #different form 1:*:* depth maps ,it will output 2:*:* mask maps

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #pose_feats is a dict:
            #key:
            """"keys
                0
                -1
                1
            """
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder_t"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    if self.opt.pose_idea == True:
                        # only estimation translation and only estimate rotation
                        translation_for_t = self.models["pose_for_t"](pose_inputs)
                        axisangle_for_r = self.models["pose_for_r"](pose_inputs)
                        outputs[("axisangle_for_r", 0, f_i)] = axisangle_for_r
                        outputs[("translation_for_t", 0, f_i)] = translation_for_t

                        # only for t
                        axisangle_temp = torch.zeros_like(axisangle_for_r)
                        outputs[("cam_T_cam_for_t", 0, f_i)] = transformation_from_parameters(
                            axisangle_temp[:, 0], translation_for_t[:, 0], invert=(f_i < 0))
                        # only for r
                        translation_temp = torch.zeros_like(translation_for_t)
                        outputs[("cam_T_cam_for_r", 0, f_i)] = transformation_from_parameters(
                            axisangle_for_r[:, 0], translation_temp[:, 0], invert=(f_i < 0))
                        # r and t
                        outputs[("cam_T_cam_r_and_t", 0, f_i)] = transformation_from_parameters(
                            axisangle_for_r[:, 0], translation_for_t[:, 0], invert=(f_i < 0))


                    translation, axisangle = self.models["pose"](pose_inputs)
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    

        """ else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":

                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                    
                    # 只预测 translation
                    pose_inputs_for_t = [self.models["pose_encoder_only_t"](torch.cat(pose_inputs, 1))]
                    # 只预测 rotation
                    pose_inputs_for_r = [self.models["pose_encoder_only_r"](torch.cat(pose_inputs, 1))]
                    
                    

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)
            # 单个的translations 和 rotation
            axisangle_only_r = self.models["pose_only_r"](pose_inputs)
            translation_only_t = self.models["pose_only_t"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("axisangle_only_r", 0, f_i)] = axisangle_only_r
                    outputs[("translation_only_t", 0, f_i)] = translation_only_t

                    axisangle_temp = torch.zeros_like(axisangle)
                    outputs[("cam_T_cam_only_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_temp[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam_r_and_t", 0, f_i)] = transformation_from_parameters(
                        axisangle_only_r[:, 0], translation_only_t[:, 0], invert=(f_i < 0))
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i]) 
            """

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            #self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp_t = outputs[("disp_t", scale)]
            disp_s = outputs[("disp", scale)]
            
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp_t = F.interpolate(
                    disp_t, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_s = F.interpolate(
                    disp_s, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth_s = disp_to_depth(disp_s, self.opt.min_depth, self.opt.max_depth)#disp_to_depth function is in layers.py
            _, depth_t = disp_to_depth(disp_t, self.opt.min_depth, self.opt.max_depth)#disp_to_depth function is in layers.py

            outputs[("depth_t", 0, scale)] = depth_t

            for _, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    if self.opt.pose_idea == True:
                        T_for_t = outputs[("cam_T_cam_for_t", 0, frame_id)]
                        T_r_and_t = outputs[("cam_T_cam_r_and_t", 0, frame_id)]
                        T_for_r = outputs[("cam_T_cam_for_r", 0, frame_id)]
                    T = outputs[("cam_T_cam", 0, frame_id)]
                # from the authors of https://arxiv.org/abs/1712.00175
                """ if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                """
                cam_points_s = self.backproject_depth[source_scale](
                    depth_s, inputs[("inv_K", source_scale)])
                
                teacher_cam_points = self.backproject_depth[source_scale](
                    depth_t, inputs[("inv_K", source_scale)])
                
                if self.opt.pose_idea == True:
                    # trasnlation
                    pix_coords_for_t = self.project_3d[source_scale](
                        cam_points_s, inputs[("K", source_scale)], T_for_t)

                    outputs[("color_for_t", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        pix_coords_for_t,
                        padding_mode="border")

                    # rotation
                    pix_coords_for_r = self.project_3d[source_scale](
                        cam_points_s, inputs[("K", source_scale)], T_for_r)

                    outputs[("color_for_r", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        pix_coords_for_r,
                        padding_mode="border")

                    # translation and rotation
                    pix_coords_r_and_t = self.project_3d[source_scale](
                        cam_points_s, inputs[("K", source_scale)], T_r_and_t)
                    
                    outputs[("color_r_and_t", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        pix_coords_r_and_t,
                        padding_mode="border")
                    
                    # teacher part
                    # trasnlation
                    teacher_pix_coords_for_t = self.project_3d[source_scale](
                        teacher_cam_points, inputs[("K", source_scale)], T_for_t)

                    outputs[("teacher_color_for_t", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        teacher_pix_coords_for_t,
                        padding_mode="border")

                    # rotation
                    teacher_pix_coords_for_r = self.project_3d[source_scale](
                        teacher_cam_points, inputs[("K", source_scale)], T_for_r)

                    outputs[("teacher_color_for_r", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        teacher_pix_coords_for_r,
                        padding_mode="border")

                    # translation and rotation
                    teacher_pix_coords_r_and_t = self.project_3d[source_scale](
                        teacher_cam_points, inputs[("K", source_scale)], T_r_and_t)

                    outputs[("teacher_color_r_and_t", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        teacher_pix_coords_r_and_t,
                        padding_mode="border")

                
                pix_coords = self.project_3d[source_scale](
                        cam_points_s, inputs[("K", source_scale)], T)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border")

                teacher_pix_coords = self.project_3d[source_scale](
                    teacher_cam_points, inputs[("K", source_scale)], T)

                outputs[("teacher_color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    teacher_pix_coords,
                    padding_mode="border")
                    
                if not self.opt.disable_automasking:
                    #doing this
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)
    
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []
            reprojection_losses_teacher = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            disp_teacher = outputs[("disp_t", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # mini reconstruction loss
            for frame_id in self.opt.frame_ids[1:]:
                
                if self.opt.pose_idea == True:
                    pred_for_t = outputs[("color_for_t", frame_id, scale)]
                    pred_r_and_t = outputs[("color_r_and_t", frame_id, scale)]
                    pred_for_r = outputs[("color_for_r", frame_id, scale)] 

                    loss_pred_for_t = self.compute_reprojection_loss(pred_for_t, target)
                    loss_pred_r_and_t = self.compute_reprojection_loss(pred_r_and_t, target)
                    loss_pred_for_r = self.compute_reprojection_loss(pred_for_r, target)

                teacher_pred_for_t = outputs[("teacher_color_for_t", frame_id, scale)]
                teacher_pred_r_and_t = outputs[("teacher_color_r_and_t", frame_id, scale)]
                teacher_pred_for_r = outputs[("teacher_color_for_r", frame_id, scale)] 

                teacher_loss_pred_for_t = self.compute_reprojection_loss(teacher_pred_for_t, target)
                teacher_loss_pred_r_and_t = self.compute_reprojection_loss(teacher_pred_r_and_t, target)
                teacher_loss_pred_for_r = self.compute_reprojection_loss(teacher_pred_for_r, target)

                pred = outputs[("color", frame_id, scale)]
                loss_pred = self.compute_reprojection_loss(pred, target)
                    
                teacher_pred = outputs[("teacher_color", frame_id, scale)]
                teacher_loss_pred = self.compute_reprojection_loss(teacher_pred, target)
                
                if self.opt.pose_idea == True:
                    final_t, _ = torch.min(torch.cat((teacher_loss_pred, teacher_loss_pred_for_t, teacher_loss_pred_r_and_t, teacher_loss_pred_for_r), 1), 1, True)
                    final_s, _ = torch.min(torch.cat((loss_pred, loss_pred_for_t, loss_pred_r_and_t, loss_pred_for_r), 1), 1, True)
                else:
                    final_t = teacher_loss_pred
                    final_s = loss_pred

                reprojection_losses.append(final_s)
                reprojection_losses_teacher.append(final_t)

            reprojection_losses = torch.cat(reprojection_losses, 1)
            reprojection_losses_teacher = torch.cat(reprojection_losses_teacher, 1)

            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                #reprojection_losses.size() =12X2X192X640 

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda()) if torch.cuda.is_available() else   0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cpu())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                #doing_this
                reprojection_loss = reprojection_losses
                reprojection_loss_teacher = reprojection_losses_teacher

            if not self.opt.disable_automasking:
                #doing_this
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cuda() * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                combined_t = torch.cat((identity_reprojection_loss, reprojection_loss_teacher), dim=1)
            else:
                combined = reprojection_loss
                combined_t = reprojection_loss_teacher

            if combined.shape[1] == 1:
                to_optimise = combined
                to_optimise_t = combined_t
            else:
                #doing this
                to_optimise, idxs = torch.min(combined, dim=1)
                to_optimise_t, idxs_t = torch.min(combined_t, dim=1)
            if not self.opt.disable_automasking:
                #outputs["identity_selection/{}".format(scale)] = (
                outputs["identity_selection/{}".format(0)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()


            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)#defualt=1e-3 something with get_smooth_loss function
            

            disp = F.interpolate(disp, scale_factor=2 ** scale, mode="nearest")
            disp_teacher = F.interpolate(disp_teacher, scale_factor=2 ** scale, mode="nearest")
            
            selective_help = to_optimise < to_optimise_t
            multiply_result = selective_help * torch.abs(disp - disp_teacher)
            loss += torch.mean(multiply_result) * (2 ** scale)
            
            student_part = selective_help * to_optimise * (2 ** scale)
            teacher_part = (~selective_help) * to_optimise_t
            two_part = student_part + teacher_part
            loss += two_part.mean()


            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss 
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so i#s only used to give an indication of validation performance


        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        #writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     try:
        #         print("Loading Adam weights")
        #         optimizer_dict = torch.load(optimizer_load_path)
        #         self.model_optimizer.load_state_dict(optimizer_dict)
        #     except Exception as e:
        #         print(f"can not load adam and wrong is {e}")
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
