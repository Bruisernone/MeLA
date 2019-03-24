import lib.config.config as cfg
import tensorflow as tf
import lib.networks.mela as mela
import lib.utils.datasets_generator as dg
import lib.utils.SaveAndLoad as Sal
import matplotlib.pyplot as plt
import numpy as np
import os

main_path = os.path.abspath('main.py')  # 获取'main.py'文件的绝对路径
project_path = os.path.dirname(main_path)  # 当前工程的目录
path_for_datasets = os.path.join(project_path, "data", "datasets.pkl")
cfg.FLAGS.path_for_graph = os.path.join(project_path, "model", "graph")
cfg.FLAGS.path_for_ckpt = os.path.join(project_path, "model", "mela.ckpt")

if os.path.isfile(path_for_datasets):
    datasets = Sal.load_pkl(path_for_datasets)
else:
    dg.datasets_generation(path_for_datasets)
    datasets = Sal.load_pkl(path_for_datasets)


network = mela.MeLA()
megred, predictions, loss, train_step = network.built_network()

# variable initializer
init = tf.global_variables_initializer()

# set config
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True

# init session
with tf.Session(config=tfconfig) as sess:
    saver = tf.train.Saver()
    if os.access(cfg.FLAGS.path_for_ckpt + ".meta", os.F_OK):
        print("continue training...")
        saver.restore(sess, cfg.FLAGS.path_for_ckpt)
        network.training(megred, datasets, sess, predictions, loss, train_step, saver)
    else:
        print("init...")
        sess.run(init)
        network.training(megred, datasets, sess, predictions, loss, train_step, saver)

pass
