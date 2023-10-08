# from typing import Set, List

import os
# import random
# import shutil
# import argparse

# import torch
# import numpy as np

# from utils.kp_diff import flip_check
# from models.Alignment import Alignment
# from models.Embedding import Embedding
# import datetime
import tornado.web
import tornado.ioloop
import signal
import json
# from base64 import b64decode,b64encode
# import hashlib
# from PIL import Image
# import time
import api



# 启动Web服务
def web_start():
    loop = tornado.ioloop.IOLoop.current()
    print('Web server starting')
    app = tornado.web.Application([
        (r"/",IndexHandler), # 首页请求路由
        (r"/upload",UploadHandler), # 上传用户图片请求路由
        (r"/swap",SwapHandler), # 换发型请求路由
        (r"/uploadimg/user",UploadUserImageHandler), # 直接通过网页上传用户图片请求路由
        (r"/uploadimg/model",UploadModelImageHandler), # 直接通过网页上传模特图片请求路由
        ],static_path='./static',template_path ='./template')
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(8000)
    # http_server.start(0)
    http_server.start(1)
    signal.signal(signal.SIGINT, sig_exit)
    print('Web server started and waiting for process')
    loop.start()

# 退出Web服务信号监听回调
def sig_exit(signum, frame):
    tornado.ioloop.IOLoop.current().add_callback_from_signal(do_stop)

# 停止Web服务回调
def do_stop():
    tornado.ioloop.IOLoop.current().stop()
    print("Web server shutdown")


# APP端接口上传用户图片请求路由处理
class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        body = json.loads(self.request.body)
        pic_base64 = body["pic"]
        pic_type = body["type"]

        #调用上传用户图片API
        result = api.upload_user_img(pic_base64,pic_type)
        self.write(result)

# APP端接口换发型请求路由处理
class SwapHandler(tornado.web.RequestHandler):
    def post(self):
        body = json.loads(self.request.body)
        source_res_id = body["userImageID"]
        target_res_id = body["targetImageID"]

        #调用换发型API
        result = api.hair_swap(source_res_id,target_res_id)
        self.write(result)

# 首页请求路由处理，非APP端访问
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello")

# 直接通过网页上传用户图片请求路由处理，非APP端访问
class UploadUserImageHandler(tornado.web.RequestHandler):
    def get(self):
        path = os.getcwd() + "/static/user_img"
        self.render('upload_file.html',path = os.listdir(path))


    def post(self):
        upload_path = './static/user_img'  # 配置文件上传的路径
        file_metas = self.request.files.get('file', [])  # 获取文件对象
        for meta in file_metas:
            file_name = meta.get('filename')
            print('uploadimg:')
            print(file_name)
            file_path = os.path.join(upload_path, file_name)  # 拼接路径
            with open(file_path, 'wb') as f:
                f.write(meta.get('body'))  # 写入内容

        return self.redirect('/uploadimg/user')

# 直接通过网页上传模特图片请求路由处理，非APP端访问
class UploadModelImageHandler(tornado.web.RequestHandler):
    def get(self):
        path = os.getcwd() + "/static/model_img"
        self.render('upload_file.html',path = os.listdir(path))


    def post(self):
        upload_path = './static/model_img'  # 配置文件上传的路径
        file_metas = self.request.files.get('file', [])  # 获取文件对象
        for meta in file_metas:
            file_name = meta.get('filename')
            file_path = os.path.join(upload_path, file_name)  # 拼接路径
            with open(file_path, 'wb') as f:
                f.write(meta.get('body'))  # 写入内容

        return self.redirect('/uploadimg/model')


if __name__ == "__main__":
    # Step 1 :执行神经网络初始化等操作
    api.net_init()

    # Step 2 :启动Web服务
    web_start()
