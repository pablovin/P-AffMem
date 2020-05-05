import PK
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

with tf.compat.v1.Session(config=config) as sess:
    pk = PK.Model(sess, useEmotion=True)

    print ("-----------")
    print ("Initializing training!")

    # with tf.device("/device:GPU:0"):
    pk.train(num_epochs=2, use_trained_model=False)
