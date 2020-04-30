"""
Implementation of the model.

Parts of the code are inherited from the official CAAE implementation (https://arxiv.org/abs/1702.08423, https://github.com/ZZUTK/Face-Aging-CAAE).
"""

import os
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat

from PK_Utils.PK_config import *
from PK_Utils.PK_image_ops import *
from PK_Utils.PK_subnetworks import encoder, generator, d_img, d_prior, d_em
from PK_Utils.PK_vgg_face import face_embedding


from metrics import concordance_cc

class Model(object):
    """
    Implementation of the model used.
    """
    def __init__(self, session):        
        self.session = session
        self.vgg_weights = loadmat(vggMat)
        
        # -- INPUT PLACEHOLDERS -----------------------------------------------------------
        # ---------------------------------------------------------------------------------
        self.input_image = tf.compat.v1.placeholder(
            tf.float32,
            [size_batch, size_image, size_image, 3],
            name='input_images'
        )

        self.valence = tf.compat.v1.placeholder(
            tf.float32,
            [size_batch, 1],
            name='valence_labels'
        )
        
        self.arousal = tf.compat.v1.placeholder(
            tf.float32,
            [size_batch, 1],
            name='arousal_labels'
        )

        self.z_prior = tf.compat.v1.placeholder(
            tf.float32,
            [size_batch, num_z_channels],
            name='z_prior'
        )

        # -- GRAPH ------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        print ('\n\t SETTING  UP THE GRAPH')

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            # with tf.device('/device:GPU:0'):
            with tf.device(device):

                # -- NETWORKS -------------------------------------------------------------
                # -------------------------------------------------------------------------

                # encoder:
                self.z = encoder(self.input_image)

                # generator: z + arousal + valence --> generated image
                self.G = generator(self.z,
                                   valence=self.valence,
                                   arousal=self.arousal)

                # discriminator on z
                self.D_z, self.D_z_logits = d_prior(self.z)

                # discriminator on G
                self.D_G, self.D_G_logits = d_img(self.G,
                                                              valence=self.valence,
                                                              arousal=self.arousal)

                # discriminator on z_prior
                self.D_z_prior, self.D_z_prior_logits = d_prior(self.z_prior,
                                                                        reuse_variables=True)

                # discriminator on input image
                self.D_input, self.D_input_logits = d_img(self.input_image,
                                                                      valence=self.valence,
                                                                      arousal=self.arousal,
                                                                      reuse_variables=True)


                # discriminator on arousal/valence
                #
                self.D_emArousal, self.D_emValence, self.D_em_arousal_logits, self.D_em_valence_logits = d_em(self.z, reuse_variables=True)

                #


            # -- LOSSES ---------------------------------------------------------------
            # -------------------------------------------------------------------------

            # ---- VGG LOSS ---------------------------------------------------------
            # The computation of this loss is inherited from the official ExprGan implementation (https://arxiv.org/abs/1709.03842, https://github.com/HuiDingUMD/ExprGAN).

            with tf.device('/device:CPU:0'):
                real_conv1_2, real_conv2_2, real_conv3_2, real_conv4_2, real_conv5_2 = face_embedding(self.vgg_weights, self.input_image)
                fake_conv1_2, fake_conv2_2, fake_conv3_2, fake_conv4_2, fake_conv5_2 = face_embedding(self.vgg_weights, self.G)

            conv1_2_loss = tf.reduce_mean(tf.abs(real_conv1_2 - fake_conv1_2)) / 224. / 224.
            conv2_2_loss = tf.reduce_mean(tf.abs(real_conv2_2 - fake_conv2_2)) / 112. / 112.
            conv3_2_loss = tf.reduce_mean(tf.abs(real_conv3_2 - fake_conv3_2)) / 56. / 56.
            conv4_2_loss = tf.reduce_mean(tf.abs(real_conv4_2 - fake_conv4_2)) / 28. / 28.
            conv5_2_loss = tf.reduce_mean(tf.abs(real_conv5_2 - fake_conv5_2)) / 14. / 14.
            self.vgg_loss = conv1_2_loss + conv2_2_loss + conv3_2_loss + conv4_2_loss + conv5_2_loss
            # -----------------------------------------------------------------------
            
            # reconstruction loss of encoder+generator
            self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

            # loss function of discriminator on z
            self.D_z_loss_prior = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_prior_logits, labels=tf.ones_like(self.D_z_prior_logits))
            )

            self.D_z_loss_z = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.zeros_like(self.D_z_logits))
            )

            self.E_z_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.ones_like(self.D_z_logits))
            )
            # loss function of discriminator on image
            self.D_img_loss_input = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_input_logits, labels=tf.ones_like(self.D_input_logits))
            )
            self.D_img_loss_G = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.zeros_like(self.D_G_logits))
            )
            self.G_img_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.ones_like(self.D_G_logits))
            )

	        # loss function of d_em on arousal and valence values
            self.D_em_loss = tf.compat.v1.losses.mean_squared_error(predictions=self.D_em_valence_logits, labels=self.valence)  + tf.compat.v1.losses.mean_squared_error(self.D_em_arousal_logits, self.arousal)


	        #CCC for arousal and valence
            self.D_em_ccc_arousal = concordance_cc(self.D_em_arousal_logits, self.arousal)
            self.D_em_ccc_valence = concordance_cc(self.D_em_valence_logits, self.valence)


            # -- TRAINABLE VARIABLES ----------------------------------------------------------
            # ---------------------------------------------------------------------------------
            trainable_variables =tf.compat.v1.trainable_variables()
            # variables of encoder
            self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
            # variables of generator
            self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
            # variables of discriminator on prior
            self.D_z_variables = [var for var in trainable_variables if 'D_prior_' in var.name]
            # variables of discriminator on realImage
            self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]
           # variables of discriminator on emotions
            self.D_em_variables = [var for var in trainable_variables if 'D_em_' in var.name]

            # -- SUMMARY ----------------------------------------------------------------------
            # ---------------------------------------------------------------------------------
            # with tf.device('/device:CPU:0'):
            self.z_summary = tf.compat.v1.summary.histogram('z', self.z)
            self.z_prior_summary = tf.compat.v1.summary.histogram('z_prior', self.z_prior)
            self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
            self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
            self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
            self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
            self.D_z_logits_summary = tf.compat.v1.summary.histogram('D_z_logits', self.D_z_logits)
            self.D_z_prior_logits_summary = tf.compat.v1.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
            self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
            self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
            self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
            self.D_G_logits_summary = tf.compat.v1.summary.histogram('D_G_logits', self.D_G_logits)
            self.D_input_logits_summary = tf.compat.v1.summary.histogram('D_input_logits', self.D_input_logits)
            self.D_em_arousal_logits_summary = tf.compat.v1.summary.histogram('D_em_arousal_logits', self.D_em_arousal_logits)
            self.D_em_valence_logits_summary = tf.compat.v1.summary.histogram('D_em_valence_logits',
                                                                              self.D_em_valence_logits)
            self.D_em_loss_summary = tf.compat.v1.summary.histogram('D_em_loss', self.D_em_loss)
            self.D_em_ccc_arousal_summary = tf.compat.v1.summary.histogram('D_em_ccc_arousal', self.D_em_ccc_arousal)
            self.D_em_ccc_valence_summary = tf.compat.v1.summary.histogram('D_em_ccc_valence', self.D_em_ccc_valence)
            self.vgg_loss_summary = tf.summary.scalar('VGG_loss', self.vgg_loss)

            # for saving the graph and variables
            self.saver = tf.compat.v1.train.Saver(max_to_keep=10)

    def train(self,
              num_epochs=2,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              use_trained_model=False,  # used the saved checkpoint to initialize the model
              ):

        enable_shuffle = True

        # set learning rate decay
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            with tf.device('/device:CPU:0'):
                self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')

        
        # -- LOAD FILE NAMES --------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        
        # ---- TRAINING DATA   
        file_names = [data_path + x for x in os.listdir(data_path)][0:10000]
        file_names = self.fill_up_equally(file_names)
        size_data = len(file_names)
        np.random.shuffle(file_names)

        # ---- VALIDATION DATA
        self.validation_files = [validation_path + v for v in os.listdir(validation_path)]


        # ---------------------------------------------------------------------------------
        self.loss_EG = self.EG_loss +  self.vgg_loss*0.3 +  0.01 * self.E_z_loss  + 0.01 * self.G_img_loss
        # self.loss_EG = self.EG_loss + self.D_em_loss * 0.02 + self.vgg_loss * 0.3 + 0.01 * self.E_z_loss + 0.01 * self.G_img_loss
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z

        # -- OPTIMIZERS -------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        with tf.device(device):
            # with tf.device('/device:GPU:0'):
            
            EG_learning_rate = tf.compat.v1.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=self.EG_global_step,
                decay_steps=size_data / size_batch * 2,
                decay_rate=decay_rate,
                staircase=True
            )

            # optimizer for encoder + generator
            self.EG_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )

            # optimizer for discriminator on z
            self.D_z_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.D_z_variables
            )

            # optimizer for discriminator on image
            self.D_img_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
            )

            # # optimizer for emotion
            # self.D_em_optimizer = tf.compat.v1.train.AdamOptimizer(
            #     learning_rate=EG_learning_rate,
            #     beta1=beta1
            # ).minimize(
            #     loss=self.D_em_loss,
            #     var_list=self.D_em_variables
            # )


        # # -- TENSORBOARD WRITER ----------------------------------------------------------
        # # ---------------------------------------------------------------------------------
        # self.writer = tf.summary.create_file_writer(save_dir)

        # -- TENSORBOARD SUMMARY ----------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # with tf.device('/device:CPU:0'):
        #     self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        #     self.summary = tf.compat.v1.summary.merge([
        #         self.z_summary, self.z_prior_summary,
        #         self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
        #         self.D_z_logits_summary, self.D_z_prior_logits_summary,
        #         self.EG_loss_summary, self.E_z_loss_summary,
        #         self.D_img_loss_input_summary, self.D_img_loss_G_summary,
        #         self.G_img_loss_summary, self.EG_learning_rate_summary,
        #         self.D_G_logits_summary, self.D_input_logits_summary,
        #         self.vgg_loss_summary, self.D_em_arousal_logits_summary, self.D_em_valence_logits_summary,  self.D_em_loss_summary, self.D_em_ccc_arousal_summary, self.D_em_ccc_valence_summary
        #     ])
        #     self.writer = tf.summary.FileWriter(os.path.join(save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:size_batch]

        file_names[0:size_batch] = []

        sample = [load_image(
            image_path=sample_file,
            image_size=size_image,
            image_value_range=image_value_range,
            is_gray=False,
        ) for sample_file in sample_files]

        sample_images = np.array(sample).astype(np.float32)


        sample_label_arousal = np.asarray([[float(x.split('__')[2])] for x in sample_files])
        sample_label_valence = np.asarray([[float(x.split('__')[3][0:-4])] for x in sample_files])

        # ******************************************* training *******************************************************
        print('\n\tPreparing for training ...')

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # epoch iteration
        num_batches = len(file_names) // size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_files = file_names[ind_batch*size_batch:(ind_batch+1)*size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=size_image,
                    image_value_range=image_value_range,
                    is_gray=False,
                ) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                batch_label_valence = np.asarray([[float(x.split('__')[2])] for x in batch_files])
                batch_label_arousal = np.asarray([[float(x.split('__')[3][0:-4])] for x in batch_files])

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    image_value_range[0],
                    image_value_range[-1],
                    [size_batch, num_z_channels]
                ).astype(np.float32)

                # # update
                # _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, vgg, em, arousalCCC, valenceCCC = self.session.run(
                #     fetches = [
                #         self.EG_optimizer,
                #         self.D_z_optimizer,
                #         self.D_img_optimizer,
                #         self.EG_loss,
                #         self.E_z_loss,
                #         self.D_z_loss_z,
                #         self.D_z_loss_prior,
                #         self.G_img_loss,
                #         self.D_img_loss_G,
                #         self.D_img_loss_input,
                #         # self.tv_loss,
                #         self.vgg_loss,
                #         self.D_em_loss,
	            #     self.D_em_ccc_arousal,
	            #     self.D_em_ccc_valence
                #     ],
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.valence: batch_label_valence,
                #         self.arousal: batch_label_arousal,
                #         self.z_prior: batch_z_prior
                #     }
                # )

                # update
                _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, vgg = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.EG_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        # self.tv_loss,
                        self.vgg_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.valence: batch_label_valence,
                        self.arousal: batch_label_arousal,
                        self.z_prior: batch_z_prior
                    }
                )
                # print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tVGG=%.4f\tEm=%.4f" %
                #     (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err, vgg, em))
                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tVGG=%.4f" %
                      (epoch + 1, num_epochs, ind_batch + 1, num_batches, EG_err, vgg))
                print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))
                # print("\tArousalCCC=%.4f\tValenceCCC=%.4f" % (arousalCCC, valenceCCC))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # # add to summary
                # summary = self.summary.eval(
                #     feed_dict={
                #         self.input_image: batch_images,
                #         self.valence: batch_label_valence,
                #         self.arousal: batch_label_arousal,
                #         self.z_prior: batch_z_prior
                #     }
                # )
                # self.writer.add_summary(summary, self.EG_global_step.eval())

                if ind_batch%500 == 0:
                    # save sample images for each epoch
                    name = '{:02d}_{:02d}'.format(epoch+1, ind_batch)
                    self.sample(sample_images, sample_label_valence, sample_label_arousal, name+'.png')
                    # TEST
                    test_dir = os.path.join(save_dir, 'test')
                    if not os.path.exists(test_dir):
                        os.makedirs(test_dir)
                    self.test(sample_images, test_dir, name+'.png')

            # save checkpoint for each epoch
            # VALIDATE
            name = '{:02d}_model'.format(epoch+1)
            self.validate(name)
            self.save_checkpoint(name=name)


    def save_checkpoint(self, name=''):
        checkpoint_dir = os.path.join(save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, name)
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, valence, arousal, name):
        sample_dir = os.path.join(save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.valence: valence,
                self.arousal: arousal
            }
        )

        size_frame = int(np.sqrt(size_batch))+1
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=image_value_range,
            size_frame=[size_frame, size_frame]
        )

        save_batch_images(
            batch_images=images,
            save_path=os.path.join(sample_dir, "input.png"),
            image_value_range=image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def validate(self, name):
        # Create Validation Directory if needed
        val_dir = os.path.join(save_dir, 'validation')
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        # Create Name Directory if needed
        name_dir = os.path.join(val_dir, name)
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)

        # validate
        testFile = self.validation_files[0:10]
        for image_path in testFile:
            n = image_path.split("/")[-1]+".png"
            self.test(np.array([load_image(image_path, image_size=96)]), name_dir, n)

    def test(self, images, test_dir, name):
        images = images[:1, :, :, :]
        # valence


        valence = np.arange(0.75, -0.751, -0.375)
        valence = np.repeat(valence, 5).reshape((25, 1))
        # valence = np.repeat(valence, 7, axis=0)
        # arousal
        arousal = [np.arange(0.75, -0.751, -0.375)]
        arousal = np.repeat(arousal, 5).reshape((25, 1))
        arousal = np.asarray([item for sublist in arousal for item in sublist]).reshape((25, 1))

        # arousal = np.repeat(arousal, 7, axis=0)
        # arousal = np.asarray([item for sublist in arousal for item in sublist]).reshape((49, 1))
        # arousal = np.asarray([item for sublist in arousal for item in sublist]).reshape((48, 1))
        query_images = np.tile(images, (25, 1, 1, 1))


        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.valence: valence,
                self.arousal: arousal
            }
        )
        save_output(
            input_image=images,
            output=G,
            path=os.path.join(test_dir, name),
            image_value_range = image_value_range
        )


    def fill_up_equally(self, X):
        # print ("Value:", X[0])
        # print ("Value:", X[0].split("s"))
        # input("here")
        sorted_samples = [[x for x in X if int(x.split('__')[1]) == r] for r in range(8)]

        amounts = [len(x) for x in sorted_samples]
        differences = [max(amounts) - a for a in amounts]

        for i, d in enumerate(differences):
            samples = sorted_samples[i]
            added = [samples[x] for x in np.random.choice(range(len(samples)), d)]
            sorted_samples[i] = sorted_samples[i] + added

        sorted_samples_flat = [item for sublist in sorted_samples for item in sublist]

        np.random.seed = 1234567
        np.random.shuffle(sorted_samples_flat)

        return sorted_samples_flat


class Logger(object):
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)

    def close(self):
        self.log.close()

    def flush(self):
        self.close()
        # needed for python 3 compatibility
        pass
