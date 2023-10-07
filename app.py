from typing import Set, List

import os
import random
import shutil
import argparse

import torch
import numpy as np

from utils.kp_diff import flip_check
from models.Alignment import Alignment
from models.Embedding import Embedding
import datetime
import tornado.web
import tornado.ioloop
import signal
import json
from base64 import b64decode,b64encode
import hashlib
from PIL import Image
import time


g_ii2s = None
g_align = None
g_args = None

def net_init():
    global g_ii2s
    global g_align
    global g_args
    args = g_args
    print('Model network init start:')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    g_ii2s = Embedding(args)
    g_align = Alignment(args)
    print('Model network init end:')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def net_start(user_img_id,target_img_id):
    global g_ii2s
    global g_align
    global g_args
    args = g_args
    print('Model network process start:')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    im_path1 = os.path.join(args.user_img_dir, user_img_id + args.img_type)
    im_path2 = os.path.join(args.model_img_dir, target_img_id + args.img_type)
    # print('im_path2:')
    # print(im_path2)

    check_has_result = check_im_paths_aligned(args,im_path1,im_path2)

    if check_has_result:
        print('has result')
        return

    ii2s = g_ii2s

    
    if args.flip_check:
        im_path2 = flip_check(im_path1, im_path2, args.device)

    # Step 1 : Embedding source and target images into W+, FS space
    im_paths_not_embedded = get_im_paths_not_embedded_W({im_path1})
    if im_paths_not_embedded:
        args.embedding_dir = args.output_dir
        ii2s.invert_images_in_W(im_paths_not_embedded)

    im_paths_not_embedded2 = get_im_paths_not_embedded_W({im_path2})
    if im_paths_not_embedded2:
        args.embedding_dir = args.output_dir
        ii2s.invert_images_in_W(im_paths_not_embedded2)

    im_paths_not_embedded3 = get_im_paths_not_embedded_FS({im_path1})
    if im_paths_not_embedded3:
        args.embedding_dir = args.output_dir
        ii2s.invert_images_in_FS(im_paths_not_embedded3)

    im_paths_not_embedded4 = get_im_paths_not_embedded_FS({im_path2})
    if im_paths_not_embedded4:
        args.embedding_dir = args.output_dir
        ii2s.invert_images_in_FS(im_paths_not_embedded4)

    if args.save_all:
        im_name_1 = os.path.splitext(os.path.basename(im_path1))[0]
        im_name_2 = os.path.splitext(os.path.basename(im_path2))[0]
        args.save_dir = os.path.join(args.output_dir, f'{im_name_1}_{im_name_2}_{args.version}')
        os.makedirs(args.save_dir, exist_ok = True)
        shutil.copy(im_path1, os.path.join(args.save_dir, im_name_1 + g_args.img_type))
        shutil.copy(im_path2, os.path.join(args.save_dir, im_name_2 + g_args.img_type))

    # Step 2 : Hairstyle transfer using the above embedded vector or tensor
    align = g_align
    align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello Tornado")

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        upload_path = './static/user_img'  # 配置文件上传的路径
        body = json.loads(self.request.body)
        pic_base64 = body["pic"]
        pic = b64decode(pic_base64)
        pic_type = body["type"]
        if pic_type is None:
            pic_type = "jpg"
        file_res_id = string_to_md5(pic_base64)
        file_name = file_res_id + "." + pic_type
        file_path = os.path.join(upload_path, file_name)  # 拼接路径
        success = False
        with open(file_path, 'wb') as f:
            f.write(pic)  # 写入内容
        if os.path.exists(file_path):
            im = Image.open(file_path)  # 打开图片
            im = im.resize((1024, 1024))  # 设置图片大小
            aa = 'jpeg'
            if pic_type == 'jpg':
                aa = 'jpeg'
            elif pic_type == 'png':
                aa = 'png'
            im.save(file_path, aa)
        success = os.path.exists(file_path)
        if success:
            self.write('{\"code\":200,\"data\":\"' +file_res_id+'\"}')
        else:
            self.write('{\"code\":400,\"data\":\"\"}')

class SwapHandler(tornado.web.RequestHandler):
    def post(self):
        global g_args
        download_path = './static/result'  # 配置文件下载的路径
        success = False
        body = json.loads(self.request.body)
        source_res_id = body["userImageID"]
        target_res_id = body["targetImageID"]
        file_url = ""
        if source_res_id and target_res_id:
            net_start(user_img_id=source_res_id,target_img_id=target_res_id)
            success = True
        sign = g_args.sign
        img_type = g_args.img_type
        # result_file = source_res_id + "_" + target_res_id + "_" + target_res_id + "_" + sign + img_type
        result_file = source_res_id + "_" + target_res_id + img_type
        download_file_path = os.path.join(download_path, result_file)  # 拼接路径
        print('download_file_path:')
        print(download_file_path)
        success = os.path.exists(download_file_path)
        result_img_data = ""
        if success:
            success = False
            with open(download_file_path,"rb") as f:
                result_img_data = b64encode(f.read())
                success = True
        if success:
            self.write('{\"code\":200,\"data\":\"' + str(result_img_data,encoding='utf-8') +'\"}')
        else:
            self.write('{\"code\":400,\"data\":\"fail\"}')

class UploadUserImageHandler(tornado.web.RequestHandler):
    def get(self):
        path = os.getcwd() + "/static/user_img"
        self.render('upload_file.html',path = os.listdir(path))


    def post(self):
        upload_path = './static/user_img'  # 配置文件上传的路径
        file_metas = self.request.files.get('file', [])  # 获取文件对象
        # [{'filename':'1.jpg', 'body': b'xxx'}, {'filename':'2.jpg', 'body': b'xxx'}]
        for meta in file_metas:  # meta {'filename':'1.jpg', 'body': b'xxx'}
            file_name = meta.get('filename')
            print('uploadimg:')
            print(file_name)
            file_path = os.path.join(upload_path, file_name)  # 拼接路径
            with open(file_path, 'wb') as f:
                f.write(meta.get('body'))  # 写入内容

        return self.redirect('/uploadimg/user')

class UploadModelImageHandler(tornado.web.RequestHandler):
    def get(self):
        path = os.getcwd() + "/static/model_img"
        self.render('upload_file.html',path = os.listdir(path))


    def post(self):
        upload_path = './static/model_img'  # 配置文件上传的路径
        file_metas = self.request.files.get('file', [])  # 获取文件对象
        # [{'filename':'1.jpg', 'body': b'xxx'}, {'filename':'2.jpg', 'body': b'xxx'}]
        for meta in file_metas:  # meta {'filename':'1.jpg', 'body': b'xxx'}
            file_name = meta.get('filename')
            file_path = os.path.join(upload_path, file_name)  # 拼接路径
            with open(file_path, 'wb') as f:
                f.write(meta.get('body'))  # 写入内容

        return self.redirect('/uploadimg/model')

def string_to_md5(string):
    md5_val = hashlib.md5(string.encode('utf8')).hexdigest()
    return md5_val

def sig_exit(signum, frame):
    tornado.ioloop.IOLoop.current().add_callback_from_signal(do_stop)

def do_stop():
    tornado.ioloop.IOLoop.current().stop()
    print("Web server shutdown")

def web_start():
    loop = tornado.ioloop.IOLoop.current()
    print('Web server starting')
    app = tornado.web.Application([
        (r"/",IndexHandler),
        (r"/upload",UploadHandler),
        (r"/uploadimg/user",UploadUserImageHandler),
        (r"/uploadimg/model",UploadModelImageHandler),
        (r"/swap",SwapHandler),
        ],static_path='./static',template_path ='./template')
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(8000)
    # http_server.start(0)
    http_server.start(1)
    signal.signal(signal.SIGINT, sig_exit)
    print('Web server started and waiting for process')
    loop.start()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_im_paths_not_embedded_W(im_paths: Set[str]) -> List[str]:
    W_embedding_dir = os.path.join(args.embedding_dir, "W+")

    im_paths_not_embedded = []
    for im_path in im_paths:
        assert os.path.isfile(im_path)

        im_name = os.path.splitext(os.path.basename(im_path))[0]
        W_exists = os.path.isfile(os.path.join(W_embedding_dir, f"{im_name}.npy"))

        if not W_exists:
            im_paths_not_embedded.append(im_path)

    return im_paths_not_embedded

def get_im_paths_not_embedded_FS(im_paths: Set[str]) -> List[str]:
    FS_embedding_dir = os.path.join(args.embedding_dir, "FS")

    im_paths_not_embedded = []
    for im_path in im_paths:
        assert os.path.isfile(im_path)

        im_name = os.path.splitext(os.path.basename(im_path))[0]
        FS_exists = os.path.isfile(os.path.join(FS_embedding_dir, f"{im_name}.npz"))

        if not FS_exists:
            im_paths_not_embedded.append(im_path)

    return im_paths_not_embedded


def check_im_paths_aligned(opts,img_path1, img_path2) -> bool:
    assert os.path.isfile(img_path1)
    assert os.path.isfile(img_path2)

    im_name_1 = os.path.splitext(os.path.basename(img_path1))[0] # source image : identity
    im_name_2 = os.path.splitext(os.path.basename(img_path2))[0] # target image : hairstyle
    file_exists = os.path.isfile(os.path.join(opts.output_dir, f'{im_name_1}_{im_name_2}{opts.img_type}'))

    return file_exists


# def main(args):

#     set_seed(42)

#     args.embedding_dir = args.output_dir

#     im_path1 = os.path.join(args.input_dir, args.im_path1)
#     im_path2 = os.path.join(args.input_dir, args.im_path2)

#     check_has_result = check_im_paths_aligned(args,im_path1,im_path2)

#     if check_has_result:
#         print('has result')
#         return

#     ii2s = Embedding(args)

    
#     if args.flip_check:
#         im_path2 = flip_check(im_path1, im_path2, args.device)

#     # Step 1 : Embedding source and target images into W+, FS space
#     im_paths_not_embedded = get_im_paths_not_embedded_W({im_path1})
#     if im_paths_not_embedded:
#         args.embedding_dir = args.output_dir
#         ii2s.invert_images_in_W(im_paths_not_embedded)

#     im_paths_not_embedded2 = get_im_paths_not_embedded_W({im_path2})
#     if im_paths_not_embedded2:
#         args.embedding_dir = args.output_dir
#         ii2s.invert_images_in_W(im_paths_not_embedded2)

#     im_paths_not_embedded3 = get_im_paths_not_embedded_FS({im_path1})
#     if im_paths_not_embedded3:
#         args.embedding_dir = args.output_dir
#         ii2s.invert_images_in_FS(im_paths_not_embedded3)

#     im_paths_not_embedded4 = get_im_paths_not_embedded_FS({im_path2})
#     if im_paths_not_embedded4:
#         args.embedding_dir = args.output_dir
#         ii2s.invert_images_in_FS(im_paths_not_embedded4)

#     if args.save_all:
#         im_name_1 = os.path.splitext(os.path.basename(im_path1))[0]
#         im_name_2 = os.path.splitext(os.path.basename(im_path2))[0]
#         args.save_dir = os.path.join(args.output_dir, f'{im_name_1}_{im_name_2}_{args.version}')
#         os.makedirs(args.save_dir, exist_ok = True)
#         shutil.copy(im_path1, os.path.join(args.save_dir, im_name_1 + '.png'))
#         shutil.copy(im_path2, os.path.join(args.save_dir, im_name_2 + '.png'))

#     # Step 2 : Hairstyle transfer using the above embedded vector or tensor
#     align = Alignment(args)
#     align.align_images(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)


def clear_all_cache_files():
    global g_args
    cache_timeout_hours = g_args.cache_timeout
    cache_timeout = cache_timeout_hours * 60 * 60
    cur_timestamp = time.time()

    # Step 1 :删除用户发型目录中的过期图片文件
    user_img_dir = g_args.user_img_dir
    for item in os.listdir(user_img_dir):
        file_path = os.path.join(user_img_dir, item)
        if not os.path.isfile(file_path):
            continue
        # print(file_path)
        ext_name = os.path.splitext(os.path.basename(file_path))[1]
        ext_name = ext_name.lower()
        # print(ext_name)
        if not (ext_name == ".jpg" or ext_name == ".png"):
            continue
        file_create_timestamp = os.path.getatime(file_path)
        # print('file_create_timestamp:')
        # print(file_create_timestamp)
        sub_timestamp = cur_timestamp - file_create_timestamp
        # print(f'sub_timestamp:{sub_timestamp}')
        if sub_timestamp < cache_timeout:
            continue
        os.remove(file_path)
        print(f'{file_path} is timeout,removed')

     
     # Step 2 :删除生成结果目录中的过期图片以及npz文件
     result_dir = g_args.output_dir



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Style Your Hair')

    # flip
    parser.add_argument('--flip_check', action='store_true', help='image2 might be flipped')

    # warping and alignment
    parser.add_argument('--warp_front_part', default=True,
                        help='optimize warped_trg img from W+ space and only optimized [:6] part')
    parser.add_argument('--warped_seg', default=True, help='create aligned mask from warped seg')
    parser.add_argument('--align_src_first', default=True, help='align src with trg mask before blending')
    parser.add_argument('--optimize_warped_trg_mask', default=True, help='optimize warped_trg_mask')
    parser.add_argument('--mean_seg', default=True, help='use mean seg when alignment')

    parser.add_argument('--kp_type', type=str, default='3D', help='kp_type')
    parser.add_argument('--kp_loss', default=True, help='use keypoint loss when alignment')
    parser.add_argument('--kp_loss_lambda', type=float, default=1000, help='kp_loss_lambda')

    # blending
    parser.add_argument('--blend_with_gram', default=True, help='add gram matrix loss in blending step')
    parser.add_argument('--blend_with_align', default=True,
                        help='optimization of alignment process with blending')


    # hair related loss
    parser.add_argument('--warp_loss_with_prev_list', nargs='+', help='select among delta_w, style_hair_slic_large',default=None)
    parser.add_argument('--sp_hair_lambda', type=float, default=5.0, help='Super pixel hair loss when embedding')


    # utils
    parser.add_argument('--version', type=str, default='v1', help='version name')
    parser.add_argument('--save_all', action='store_true',help='save all output from whole process')
    parser.add_argument('--embedding_dir', type=str, default='./output/', help='embedding vector directory')

    # I/O arguments
    # parser.add_argument('--input_dir', type=str, default='./image/',
    #                     help='The directory of the images to be inverted')
    parser.add_argument('--model_img_dir', type=str, default='./static/model_img',
                        help='The directory of the model images')
    parser.add_argument('--user_img_dir', type=str, default='./static/user_img',
                        help='The directory of the user images to be inverted')
    parser.add_argument('--output_dir', type=str, default='./static/result',
                        help='The directory to save the output images')
    parser.add_argument('--im_path1', type=str, default='16.png', help='Identity image')
    parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')
    parser.add_argument('--img_type', type=str, default='.jpg', help='image type,.png or .jpg')
    parser.add_argument('--cache_timeout', type=int, default=24, help='all cache files will be removed when timeout by hours')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
    parser.add_argument('--verbose', action='store_true', help='Print loss information')
    parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')

    # Embedding loss options
    parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
    parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
    parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
    parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
    # parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
    # parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')
    # parser.add_argument('--W_steps', type=int, default=500, help='Number of W space optimization steps')
    # parser.add_argument('--FS_steps', type=int, default=100, help='Number of W space optimization steps')
    # parser.add_argument('--W_steps', type=int, default=200, help='Number of W space optimization steps')
    # parser.add_argument('--FS_steps', type=int, default=100, help='Number of W space optimization steps')
    parser.add_argument('--W_steps', type=int, default=80, help='Number of W space optimization steps')
    parser.add_argument('--FS_steps', type=int, default=50, help='Number of W space optimization steps')

    # Alignment loss options
    parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
    parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
    # parser.add_argument('--align_steps1', type=int, default=400, help='')
    # parser.add_argument('--align_steps2', type=int, default=100, help='')
    # parser.add_argument('--warp_steps', type=int, default=100, help='')
    # parser.add_argument('--align_steps1', type=int, default=100, help='')
    # parser.add_argument('--align_steps2', type=int, default=30, help='')
    # parser.add_argument('--warp_steps', type=int, default=30, help='')
    parser.add_argument('--align_steps1', type=int, default=50, help='')
    parser.add_argument('--align_steps2', type=int, default=20, help='')
    parser.add_argument('--warp_steps', type=int, default=20, help='')

    # Blend loss options
    parser.add_argument('--face_lambda', type=float, default=3.0, help='')
    parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
    # parser.add_argument('--blend_steps', type=int, default=400, help='')
    # parser.add_argument('--blend_steps', type=int, default=100, help='')
    # parser.add_argument('--blend_steps', type=int, default=80, help='')
    parser.add_argument('--blend_steps', type=int, default=50, help='')

    args = parser.parse_args()
    # main(args)
    g_args = args
    set_seed(42)
    args.embedding_dir = args.output_dir
    args.warp_loss_with_prev_list = "delta_w style_hair_slic_large"

    clear_all_cache_files()

    net_init()

    '''TODO:test'''
    args.img_type = ".jpg"
    args.W_steps = 8
    args.FS_steps = 5
    args.align_steps1 = 5
    args.align_steps2 = 2
    args.warp_steps = 2
    args.blend_steps = 5
    # net_start(user_img_id='00016',target_img_id='00587')
    net_start(user_img_id='40096',target_img_id='00587')

    web_start()
