import PK
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

with tf.Session(config=config) as sess:
    pk = PK.Model(sess)

    print ("-----------")
    print ("Initializing training!")

    with tf.device('/device:GPU:0'):
        pk.train(num_epochs=2)
