#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     : 2022.01
# @Author   : 绿色羽毛
# @Email    : lvseyumao@foxmail.com
# @Blog     : https://blog.csdn.net/ViatorSun
# @Paper    : 
# @arXiv    : 
# @version  : "1.0" 
# @Note     : 
# 
#



import os
import logging



def create_logger(log_dir, filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    formatter = logging.Formatter(  )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])


    fh = logging.FileHandler(os.path.join(log_dir, str(filename)+'.txt'), mode='a')

    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

