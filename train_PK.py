import PK
import tensorflow as tf

from PK_Utils.PK_config import device

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

with tf.Session(config=config) as sess:
    pk = PK.Model(sess)

    print ("-----------")
    print ("Initializing training!")

    with tf.device(device):
        pk.train(num_epochs=2)
