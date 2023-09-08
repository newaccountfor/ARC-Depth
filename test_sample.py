from __future__ import absolute_import, division, print_function

import cv2
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time
import torch
from torchvision import transforms, datasets
from cv2 import imwrite
import networks
# import hr_networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_folder',type = str,
                        help='the folder name of model')
    # parser.add_argument('--model_folder_teacher', type=str, help="model_folder_teacher")
    parser.add_argument('--model_name',type = str)
    parser.add_argument('--data_path',type = str)
    '''parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                                "mono+stereo_no_pt_640x192",
                                "mono_1024x320",
                                "stereo_1024x320",
                                "mono+stereo_1024x320"])'''
    parser.add_argument('--ext', type=str,
                            help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                            help='if set, disables CUDA',
                            action='store_true')
    return parser.parse_args()

def test_simple(args):
    """Function to predict for a single image or folder of images
        """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
            #device = torch.device("cuda")
            device = "cuda"
    else:
            device = "cpu"

    model_path = os.path.join(args.model_folder, args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.test_hr_encoder.hrnet18(False)
    encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()


    # teacher network
    # encoder_teacher_path = os.path.join(args.load_weights_folder_teacher, "encoder_t.pth")
    # decoder_teacher_path = os.path.join(args.load_weights_folder_teacher, "depth_t.pth")
    # encoder_teacher_dict = torch.load(encoder_teacher_path) if torch.cuda.is_available() else torch.load(encoder_teacher_path,map_location = 'cpu')
    # decoder_teacher_dict = torch.load(decoder_teacher_path) if torch.cuda.is_available() else torch.load(decoder_teacher_path,map_location = 'cpu')
    
    # encoder_teacher = networks.ResnetEncoder(50, "pretrained", num_input_images=4)
    # decoder_teacher = networks.TeacherDecoder(encoder_teacher.num_ch_enc, 1)

    # model_dict_teacher = encoder_teacher.state_dict()
    # dec_model_dict_teacher = decoder_teacher.state_dict()
    # encoder_teacher.load_state_dict({k: v for k, v in encoder_teacher_dict.items() if k in model_dict_teacher})
    # decoder_teacher.load_state_dict({k: v for k, v in decoder_teacher_dict.items() if k in dec_model_dict_teacher})

    # encoder_teacher.cuda() if torch.cuda.is_available() else encoder_teacher.cpu()
    # encoder_teacher.eval()

    # decoder_teacher.cuda() if torch.cuda.is_available() else decoder_teacher.cpu()
    # decoder_teacher.eval()

    para_sum_encoder = sum(p.numel() for p in encoder.parameters())
    
    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc, range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    timing = 0.0
    para_sum_decoder = sum(p.numel() for p in depth_decoder.parameters())
    depth_decoder.to(device)
    depth_decoder.eval()
    para_sum = para_sum_decoder + para_sum_encoder
    print("encoder has {} parameters".format(para_sum_encoder))
    print("depth_decoder has {} parameters".format(para_sum_decoder))
    print("encoder and depth_ decoder have  total {} parameters".format(para_sum))
    
    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        f = open(args.image_path)
        f_lines = f.readlines()
        paths = f_lines
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format('png')))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue
            # Load image and preprocess
            image_path = os.path.join(args.data_path, image_path).rstrip('\n')
            input_image = pil.open(image_path).convert('RGB')
            image_name =  "-".join(image_path.split('/')[5:]).rstrip(".png")
            image_names = image_path.split('/')[6]
            rgb = transforms.ToTensor()(input_image)
            
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_rgb = input_image
            input_r = pil.fromarray(np.uint8(input_rgb))
            
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            
            rgb1 = rgb.permute(1,2,0).detach().cpu().numpy() * 255
            
            
            features = encoder(input_image)
            start_time = time.time()
            outputs = depth_decoder(features)
            end_time = time.time()
            timing += end_time - start_time

            disp = outputs[("disp", 0)]
            #disp_resized = disp
            # just like Featdepth
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            # name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, depth_resized = disp_to_depth(disp, 0.1, 100)
            # np.save(name_dest_npy, scaled_disp.cpu().numpy())

            depth_resized = torch.nn.functional.interpolate(
                depth_resized, (original_height, original_width), mode="bilinear", align_corners=False)
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            depth_resized_np = depth_resized.squeeze().cpu().numpy()

            
            gt_dir = "/home/sdb1/ouyuxiang/biaobiaobiao/gt_new"
            #save_path ="/home/sdb1/ouyuxiang/biaobiaobiao/other_depth/monodepth2-master/monodepth2_result_new"
            #save_path = "/home/sdb1/ouyuxiang/biaobiaobiao/other_depth/results/featdepth"
            gt_path = os.path.join(gt_dir, image_names)
            gt = cv2.imread(gt_path,0)
            #gt = cv2.resize(gt, (640,192))
            #gt_tensor = torch.tensor(gt).float().cuda()
            #gt_depth = gt_tensor.cpu().numpy().squeeze()

            MIN_DEPTH = 1e-3
            MAX_DEPTH = 80
            gt_height, gt_width = gt.shape[:2]
            mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            #print(depth_resized_np.shape)

            depth_resized_np_m = depth_resized_np*mask
            depth_resized_np = depth_resized_np[mask]
            gt_m= gt*mask
            gt= gt[mask]
            #pred_depth *= opt.pred_depth_scale_factor
            ratios = []
            ratio = np.median(gt) / np.median(depth_resized_np)
            ratios.append(ratio)
            depth_resized_np *= ratio
            depth_resized_np_m *= ratio

            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

            depth_resized_np[depth_resized_np < MIN_DEPTH] = MIN_DEPTH
            depth_resized_np[depth_resized_np > MAX_DEPTH] = MAX_DEPTH
            
            depth_resized_np_m[depth_resized_np_m < MIN_DEPTH] = MIN_DEPTH
            depth_resized_np_m[depth_resized_np_m > MAX_DEPTH] = MAX_DEPTH
           
        
            diff_image = np.absolute(gt_m - depth_resized_np_m)
            #diff_image_np = diff_image.squeeze().cpu().numpy()
            normalizer = mpl.colors.Normalize(vmin=MIN_DEPTH, vmax=10)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='turbo')
            colormapped_im = (mapper.to_rgba(diff_image)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            new_image_name = image_names.split(".")[0]
            name_dest_im = os.path.join(args.save_path,"{}_diff_image_320.jpeg".format(new_image_name))
            im.save(name_dest_im)

            rmse = (depth_resized_np - gt) ** 2
            rmse = np.sqrt(rmse.mean())
            print(rmse)

            file = open(args.save_path+"/rmse.txt", "a")
            file.write('{},{},{}'.format(new_image_name,rmse,"\n"))
            file.close()
            
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            name_dest_im = os.path.join(args.save_path,"{}_disp.jpeg".format(image_name))
            
            # concatenate both vertically
            #image = np.concatenate([rgb1, im], 0)
            # save a grey scale map for point cloud viz
            #depth_resized = depth_resized.squeeze().cpu().numpy()
            '''
            scaled_disp = (50 / scaled_disp).squeeze().cpu().numpy()
            #scaled_disp = scaled_disp.squeeze().cpu().numpy()
            im_grey = pil.fromarray(np.uint8((scaled_disp * 255)),'L')
            name_grey_depth = os.path.join(args.save_path,"{}_grey_disp.png".format(image_name))
            name_corped_rgb = os.path.join(args.save_path,"rgb.png")
            im_grey.save(name_grey_depth) 
            input_r.save(name_corped_rgb)
            '''
            #just save a single depth
            im.save(name_dest_im)

            #save a concatenated iamge for depth and rgb
            #imwrite(name_dest_im,image[:,:,::-1])
            
            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print(timing/32)
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
