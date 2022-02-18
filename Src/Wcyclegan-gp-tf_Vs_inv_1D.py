#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 02/18/2022

@author: aocai (aocai166@gmail.com) Rice University
"""
# Cycle GAN with Wasserstein metric and gradient penalty (Wasserstein Cycle-GAN)
# using tensorflow for deriving Vs model from dispersion curve
# Labeled data: Extracted from Community velocity model of Shaw et al. (CVMH) 16480 dispersion curves
# The synthetic dispersion data is calculated using Herrmann (2013) package
# Unlabeled data: 4076 dispersion curves from Qiu et al. 2019

from __future__ import division
import os
import sys
import time
import math
import datetime
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from collections import namedtuple
from keras.layers.merge import _Merge
from functools import partial
from Vs_inv_modules1Dkeras import *
from Vs_inv_data_loader import DataLoader


def random_interpolate(real_img, fake_img, batch_size):
    alpha = tf.keras.backend.random_uniform((batch_size, 1, 1))
    return (alpha * real_img) + ((1-alpha) * fake_img)


def balance_normalization(array, gvalmax, gvalmin):
    return 2.0 * (array-gvalmin)/(gvalmax-gvalmin) - 1.0


def balance_norm_reverse(array, gvalmax, gvalmin):
    return gvalmin + (array + 1.0) * (gvalmax-gvalmin)/2.0


class CycleGAN():
    def __init__(self, sess):
        # Input shape
        self.sess = sess
        self.disp_dim = 17
        self.vs_dim = 99
        self.nlabel = 16480
        self.ulabel = 4076
        self.Lbatch_rate = 100
        self.Ubatch_rate = 50
        self.Lsize = self.nlabel // self.Lbatch_rate
        self.Usize = self.ulabel // self.Ubatch_rate
        self.disp_channels = 2
        self.vs_channels = 1
        self.disp_shape = (self.disp_dim, self.disp_channels)
        self.vs_shape = (self.vs_dim, self.vs_channels)
        self.disp_max = 10
        self.vs_max = 10
        self.disp_min = 1
        self.vs_min = 1
        self.lr = 0.00005
        self.data_normalize = True
        self.balance_norm = True
        self.gen_archi = 'dcnn'
        self.cycle_loss = True
        self.est_loss = True
        self.cycle_mse = True
        self.est_mse = True
        self.ncritics = 5
        self.gp_weight = 100

        # Path to the folder of labeled data (e.g., disp_region/)
        self.file_train_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Train_dat/'
        # Path to the folder of labeled data (not used as the test data is a numpy file)
        self.test_disp_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/linear_test/'
        # Path to the directory where you want to store the results
        self.out_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Final_results/output_region_chi/'
        # type of test data, vary between txt and npy
        self.test_dtype = 'npy'
        self.save_freq = 25
        
        # Configure data Loader
        self.data_loader = DataLoader(file_train_path=self.file_train_path, test_disp_path=self.test_disp_path,
                                      out_path=self.out_path, nbatch_label=self.Lbatch_rate,
                                      nbatch_ulabel=self.Ubatch_rate,
                                      ntrain=self.nlabel, ntest=self.ulabel,
                                      vs_sample=self.vs_dim, disp_sample=self.disp_dim)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32
        
        # Loss weights
        self.lambda_cycle = 5.0
        self.lambda_est = 0.6 * self.lambda_cycle

        self.discriminatorP = discriminatorP(input_shape=self.vs_shape, df=self.df, name='discriminatorP')
        self.discriminatorG = discriminatorG(input_shape=self.disp_shape, df=self.df, name='discriminatorG')
        print(self.discriminatorP.summary())
        print(self.discriminatorG.summary())

        # Initialize cycle-gan structures
        if self.gen_archi == 'resnet':
            self.generatorP2G = generator_resnet_P2G(input_shape=self.vs_shape, gf=self.gf, name='generatorP2G')
            self.generatorG2P = generator_resnet_G2P(input_shape=self.disp_shape, gf=self.gf, name='generatorG2P')
        else:
            self.generatorP2G = generator_dcnn_P2G(input_shape=self.vs_shape, gf=self.gf, name='generatorP2G')
            self.generatorG2P = generator_dcnn_G2P(input_shape=self.disp_shape, gf=self.gf, name='generatorG2P')
        print(self.generatorP2G.summary())
        print(self.generatorG2P.summary())

        self.criterionGAN = wasserstein_loss
        
        if self.cycle_loss:
            if self.cycle_mse:
                self.criterion_cyc = mse_criterion
            else:
                self.criterion_cyc = mae_criterion
            
        if self.est_loss:
            if self.est_mse:
                self.criterion_est = mse_criterion
            else:
                self.criterion_est = mae_criterion
            
        self._build_model()
        self.saver = tf.train.Saver()
    
    def gradient_penalty_loss(self, y_pred, averaged_samples):
        """ Computes gradient penalty based on prediction and weighted real / fake samples """
        gradients = tf.keras.backend.gradients(y_pred, averaged_samples)[0]
        # Compute the Euclidean norm by squaring
        gradients_sqr = tf.keras.backend.square(gradients)
        # ... Summing over the rows ...
        gradients_sqr_sum = tf.keras.backend.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # ... and sqrt
        gradient_l2_norm = tf.keras.backend.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.keras.backend.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch examples
        return tf.keras.backend.mean(gradient_penalty)

    def _build_model(self):

        # Labeled impedance data
        self.real_Limp = tf.placeholder(tf.float32, [None, self.vs_dim, self.vs_channels], name="real_vs_label")
        # Labeled image data
        self.real_Limg = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="real_disp_label")
        # Unlabeled image data
        self.real_Uimg = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="real_disp_ulabel")

        # When we train the generator, we freeze the discriminator
        self.generatorG2P.trainable = True
        self.generatorP2G.trainable = True
        self.discriminatorP.trainable = False
        self.discriminatorG.trainable = False

        self.fake_Limp = self.generatorG2P(self.real_Limg)
        self.reconstr_Limg = self.generatorP2G(self.fake_Limp)
        self.fake_Limg = self.generatorP2G(self.real_Limp)
        self.reconstr_Limp = self.generatorG2P(self.fake_Limg)
        self.fake_Uimp = self.generatorG2P(self.real_Uimg)
        self.reconstr_Uimg = self.generatorP2G(self.fake_Uimp)

        # Discriminator of Labeled generated image
        self.DP_fakeL = self.discriminatorP(self.fake_Limp)
        # Discriminator of Unlabeled generated impedance
        self.DP_fakeU = self.discriminatorP(self.fake_Uimp)
        # Discriminator of Labeled generated image
        self.DG_fake = self.discriminatorG(self.fake_Limg)

        # Construct the generator loss function
        self.g_loss_g2p = self.criterionGAN(self.DP_fakeL, -tf.ones_like(self.DP_fakeL)) \
                        + self.criterionGAN(self.DP_fakeU, -tf.ones_like(self.DP_fakeU))
        if self.cycle_loss:
            self.g_loss_g2p += self.lambda_cycle * self.criterion_cyc(self.real_Limg, self.reconstr_Limg) \
                             + self.lambda_cycle * self.criterion_cyc(self.real_Uimg, self.reconstr_Uimg) \
                             + self.lambda_cycle * self.criterion_cyc(self.real_Limp, self.reconstr_Limp)
        if self.est_loss:
            self.g_loss_g2p += self.lambda_est * self.criterion_est(self.real_Limp, self.fake_Limp)

        self.g_loss_p2g = self.criterionGAN(self.DG_fake, -tf.ones_like(self.DG_fake))
        if self.cycle_loss:
            self.g_loss_p2g += self.lambda_cycle * self.criterion_cyc(self.real_Limg, self.reconstr_Limg) \
                             + self.lambda_cycle * self.criterion_cyc(self.real_Uimg, self.reconstr_Uimg) \
                             + self.lambda_cycle * self.criterion_cyc(self.real_Limp, self.reconstr_Limp)
        if self.est_loss:
            self.g_loss_p2g += self.lambda_est * self.criterion_est(self.real_Limg, self.fake_Limg)

        self.g_loss = self.criterionGAN(self.DP_fakeL, -tf.ones_like(self.DP_fakeL)) \
                    + self.criterionGAN(self.DP_fakeU, -tf.ones_like(self.DP_fakeU)) \
                    + self.criterionGAN(self.DG_fake, -tf.ones_like(self.DG_fake))
        if self.cycle_loss:
            self.g_loss += self.lambda_cycle * self.criterion_cyc(self.real_Limg, self.reconstr_Limg) \
                         + self.lambda_cycle * self.criterion_cyc(self.real_Uimg, self.reconstr_Uimg) \
                         + self.lambda_cycle * self.criterion_cyc(self.real_Limp, self.reconstr_Limp)
        if self.est_loss:
            self.g_loss += self.lambda_est * self.criterion_est(self.real_Limp, self.fake_Limp) \
                         + self.lambda_est * self.criterion_est(self.real_Limg, self.fake_Limg)

        # Generated images from labeled real impedance
        self.fake_Limg_sample = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="fake_disp_label")
        # Generated impedance from labeled real images
        self.fake_Limp_sample = tf.placeholder(tf.float32, [None, self.vs_dim, self.vs_channels], name="fake_vs_label")
        # Generated impedance from unlabeled real images
        self.fake_Uimp_sample = tf.placeholder(tf.float32, [None, self.vs_dim, self.vs_channels], name="fake_vs_ulabel")

        # When we train the discriminator, we freeze the generator instead
        self.generatorG2P.trainable = False
        self.generatorP2G.trainable = False
        self.discriminatorP.trainable = True
        self.discriminatorG.trainable = True

        # Construct weighted average between real and fake impedance
        self.interpolate_imp = random_interpolate(real_img=self.real_Limp,
                                                  fake_img=self.fake_Limp_sample,
                                                  batch_size=self.Lsize)
        # Determine validity of weighted sample
        self.validityP_interpolate = self.discriminatorP(self.interpolate_imp)

        # Use Python partial to provide loss function with additional averaged_samples to argument
        self.partial_gp_lossP = self.gradient_penalty_loss(y_pred=self.validityP_interpolate,
                                                           averaged_samples=self.interpolate_imp)
        self.partial_gp_lossP.__name__ = 'gradient_penalty_P'

        # Construct weighed average between real and fake image
        self.interpolate_img = random_interpolate(real_img=self.real_Limg,
                                                  fake_img=self.fake_Limg_sample,
                                                  batch_size=self.Lsize)

        # Deterine validity of weighted sample
        self.validityG_interpolate = self.discriminatorG(self.interpolate_img)

        # Use Python partial to provide loss funciton with additional averaged_samples to argument
        self.partial_gp_lossG = self.gradient_penalty_loss(y_pred=self.validityG_interpolate,
                                                           averaged_samples=self.interpolate_img)
        self.partial_gp_lossG.__name__ = 'gradient_penalty_G'

        # Discriminator of Labeled true impedance
        self.DP_real = self.discriminatorP(self.real_Limp)
        # Discriminator of Labeled true image
        self.DG_realL = self.discriminatorG(self.real_Limg)
        # Discriminator of Unlabeled true image
        self.DG_realU = self.discriminatorG(self.real_Uimg)
        # Discriminator of Labeled generated impedance
        self.DP_fake_sampleL = self.discriminatorP(self.fake_Limp_sample)
        # Discriminator of Unlabeled generated impedance
        self.DP_fake_sampleU = self.discriminatorP(self.fake_Uimp_sample)
        # Discriminator of Labeled generated image
        self.DG_fake_sample = self.discriminatorG(self.fake_Limg_sample)

        # Construct the discriminator loss funcitons
        self.dp_loss_real = self.criterionGAN(self.DP_real, -tf.ones_like(self.DP_real))
        self.dp_loss_fake = self.criterionGAN(self.DP_fake_sampleL, tf.ones_like(self.DP_fake_sampleL)) \
                          + self.criterionGAN(self.DP_fake_sampleU, tf.ones_like(self.DP_fake_sampleU))
        self.dp_loss = self.dp_loss_real + self.dp_loss_fake + self.gp_weight * self.partial_gp_lossP

        self.dg_loss_real = self.criterionGAN(self.DG_realL, -tf.ones_like(self.DG_realL)) \
                          + self.criterionGAN(self.DG_realU, -tf.ones_like(self.DG_realU))
        self.dg_loss_fake = self.criterionGAN(self.DG_fake_sample, tf.ones_like(self.DG_fake_sample))
        self.dg_loss = self.dg_loss_real + self.dg_loss_fake + self.gp_weight * self.partial_gp_lossG
        self.d_loss = self.dg_loss + self.dp_loss

        # Construct the tensorflow summary for the variables
        self.g_loss_g2p_sum = tf.summary.scalar("g_loss_g2p", self.g_loss_g2p)
        self.g_loss_p2g_sum = tf.summary.scalar("g_loss_p2g", self.g_loss_p2g)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_g2p_sum, self.g_loss_p2g_sum, self.g_loss_sum])

        self.dp_loss_sum = tf.summary.scalar("dp_loss", self.dp_loss)
        self.dg_loss_sum = tf.summary.scalar("dg_loss", self.dg_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.dp_loss_real_sum = tf.summary.scalar("dp_loss_real", self.dp_loss_real)
        self.dp_loss_fake_sum = tf.summary.scalar("dp_loss_fake", self.dp_loss_fake)
        self.dg_loss_real_sum = tf.summary.scalar("dg_loss_real", self.dg_loss_real)
        self.dg_loss_fake_sum = tf.summary.scalar("dg_loss_fake", self.dg_loss_fake)
        self.d_sum = tf.summary.merge(
                [self.dg_loss_sum, self.dg_loss_real_sum, self.dg_loss_fake_sum,
                 self.dp_loss_sum, self.dp_loss_real_sum, self.dp_loss_fake_sum,
                 self.d_loss_sum]
                )

        self.test_G = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="test_disp")
        self.testP = self.generatorG2P(self.test_G)
        self.reconstrG = self.generatorP2G(self.testP)

        self.test_P = tf.placeholder(tf.float32, [None, self.vs_dim, self.vs_channels], name="test_vs")
        self.testG = self.generatorP2G(self.test_P)
        self.reconstrP = self.generatorG2P(self.testG)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, epochs, batch_size=10, sample_interval=1, startfrombeg=True):
        """ Train Wasserstein Cycle GAN """

        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        #self.d_optim = tf.train.RMSPropOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        #self.g_optim = tf.train.RMSPropOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.out_path+"logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if not startfrombeg:
            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed ...")

        gan_loss_val = []

        Ldisp_data_orig, Lvs_data_orig, Udisp_data_orig, self.disp_max, self.vs_max, self.disp_min, self.vs_min,\
            self.file_name_train, self.file_name_test = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.disp_min, self.vs_max, self.vs_min)

        # Normalize the data
        if self.data_normalize:
            if self.balance_norm:
                Ldisp_data_orig = balance_normalization(Ldisp_data_orig, self.disp_max, self.disp_min)
                Lvs_data_orig = balance_normalization(Lvs_data_orig, self.vs_max, self.vs_min)
                Udisp_data_orig = balance_normalization(Udisp_data_orig, self.disp_max, self.disp_min)
            else:
                Ldisp_data_orig = Ldisp_data_orig/self.disp_max
                Lvs_data_orig = Lvs_data_orig/self.vs_max
                Udisp_data_orig = Udisp_data_orig/self.disp_max

        print(np.max(Ldisp_data_orig), np.min(Ldisp_data_orig), np.max(Lvs_data_orig), np.min(Lvs_data_orig),
              np.max(Udisp_data_orig), np.min(Udisp_data_orig))

        for epoch in range(epochs):
            if epoch == 0:
                self.sample_vsprofile(epoch)
                loss_val = self.label_vsprofile(epoch)

            #idx = np.random.permutation(self.ulabel)
            #idx = np.random.randint(self.ulabel, size=self.ulabel)

            nbatch = max(self.Lbatch_rate, self.Ubatch_rate)

            for ibatch in range(nbatch):
                Lkbatch = ibatch % self.Lbatch_rate
                Ukbatch = ibatch % self.Ubatch_rate
                if Lkbatch == 0:
                    Lidx = np.random.permutation(self.nlabel)
                if Ukbatch == 0:
                    Uidx = np.random.permutation(self.ulabel)

                # Batch the labeled data and permute
                Lleft = Lkbatch * self.Lsize
                Lright = (Lkbatch+1) * self.Lsize
                Lidx_batch = Lidx[Lleft: Lright]
                Ldisp_data = Ldisp_data_orig[Lidx_batch]
                Lvs_data = Lvs_data_orig[Lidx_batch]

                # Batch the unlabeled data and permute
                Uleft = Ukbatch * self.Usize
                Uright = (Ukbatch + 1) * self.Usize
                Uidx_batch = Uidx[Uleft:Uright]
                Udisp_data = Udisp_data_orig[Uidx_batch]

                # Update G network and record fake outputs
                fake_Lvs, fake_Ldisp, fake_Uvs, _, summary_str = self.sess.run(
                        [self.fake_Limp, self.fake_Limg, self.fake_Uimp, self.g_optim, self.g_sum],
                        feed_dict={self.real_Limg: Ldisp_data, self.real_Limp: Lvs_data, self.real_Uimg: Udisp_data})

                self.writer.add_summary(summary_str, counter)

                # Update D network
                for k in range(self.ncritics):
                    _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                                   feed_dict={self.real_Limg: Ldisp_data,
                                                              self.real_Limp: Lvs_data,
                                                              self.real_Uimg: Udisp_data,
                                                              self.fake_Limp_sample: fake_Lvs,
                                                              self.fake_Limg_sample: fake_Ldisp,
                                                              self.fake_Uimp_sample: fake_Uvs})

                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d/%2d] Lbatch: [%4d/%4d] Ubatch: [%4d/%4d] Loss:%4.6f time: %4.4f" % (
                        epoch, epochs, Lkbatch, self.Lbatch_rate, Ukbatch,
                        self.Ubatch_rate, loss_val, time.time() - start_time)))

            if epoch % self.save_freq == 0:
                self.save(epoch)

            if epoch % sample_interval == 0:
                self.sample_vsprofile(epoch)
                loss_val = self.label_vsprofile(epoch)
                gan_loss_val.append((epoch, loss_val))
                np.savetxt(self.out_path + 'Loss_func.txt', gan_loss_val)
                if loss_val <= 0.065 or np.isnan(loss_val):
                    sys.exit(0)

    def save(self, step):
        model_name = "Wcyclegan.model"
        model_dir = self.out_path
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.saver.save(self.sess,
                        os.path.join(model_dir, model_name),
                        global_step=step)
        
    def load(self):
        print("[*] Reading checkpoint ...")
        
        model_dir = self.out_path
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess,
                               os.path.join(model_dir, ckpt_name))
            return True
        else:
            return False

    def predict_vol(self, direction='G2P'):
        """ Generate predicted full seismic vs model and reconstructed dispersion curve """

        nbatch = self.Ubatch_rate // 2
        ntrace = self.ulabel
        bsize = self.ulabel // nbatch
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        _, _, _, self.disp_max, self.vs_max, self.disp_min, self.vs_min, self.file_name_train, self.file_name_test \
            = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.vs_max, self.disp_min, self.vs_min)

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed ...")

        test_full = self.data_loader.load_full_test_data()
        disp_full = test_full[:, :, 3:5]

        if self.data_normalize:
            if self.balance_norm:
                disp_full = balance_normalization(disp_full, self.disp_max, self.disp_min)
            else:
                disp_full = disp_full / self.disp_max

        fake_vs = np.zeros((disp_full.shape[0], self.vs_dim, disp_full.shape[2]-1))
        reconstr_disp = np.zeros(test_full.shape)
        reconstr_disp[:, :, 0:3] = test_full[:, :, 0:3]

        for ibatch in range(nbatch + 1):
            print("Predicting ibatch %d/%d" % (ibatch, nbatch))
            if ibatch == nbatch:
                left = ibatch * bsize
                right = ntrace - 1
            else:
                left = ibatch * bsize
                right = (ibatch + 1) * bsize
            disp_batch = disp_full[left:right]

            fake_vs_batch, reconstr_disp_batch = self.sess.run(
                [self.testP, self.reconstrG],
                feed_dict={self.test_G: disp_batch})

            fake_vs[left:right, :, :] = fake_vs_batch
            reconstr_disp[left:right, :, 3:5] = reconstr_disp_batch

        if self.data_normalize:
            if self.balance_norm:
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                reconstr_disp[:, :, 3:5] = balance_norm_reverse(reconstr_disp[:, :, 3:5], self.disp_max, self.disp_min)
            else:
                fake_vs = fake_vs * self.vs_max
                reconstr_disp[:, :, 3:5] = reconstr_disp[:, :, 3:5] * self.disp_max

        np.save(self.out_path + 'predict/' + 'Predict_vs.npy', fake_vs)
        np.save(self.out_path + 'predict/' + 'Reconstr_disp.npy', reconstr_disp)

    def test(self, direction='G2P', drawline=False):
        """ Test Cycle GAN """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        _, _, _, self.disp_max, self.vs_max, self.disp_min, self.vs_min, self.file_name_train, self.file_name_test\
            = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.vs_max, self.disp_min, self.vs_min)

        if self.load():
            print("[*] Load SUCCESS")
        else:
            print("[!] Load failed ...")

        if drawline:
            self.line_vsprofile(0)

        c = 6

        if direction == 'G2P':
            titles = ['Labeled $V_{phase}$', 'Labeled $V_{group}$', 'Translated Vs', 'Labeled Vs', 'Reconstructed $V_{phase}$',
                      'Reconstructed $V_{group}$']
            disp_select, vs_select = self.data_loader.load_label_data(nselect=100, sample=True)

            if self.data_normalize:
                if self.balance_norm:
                    disp_select = balance_normalization(disp_select, self.disp_max, self.disp_min)
                else:
                    disp_select = disp_select / self.disp_max

            fig, axs = plt.subplots(1, c, figsize=(36, 14))

            fake_vs, reconstr_disp = self.sess.run(
                [self.testP, self.reconstrG],
                feed_dict={self.test_G: disp_select})

            if self.data_normalize:
                if self.balance_norm:
                    disp_select = balance_norm_reverse(disp_select, self.disp_max, self.disp_min)
                    fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                    reconstr_disp = balance_norm_reverse(reconstr_disp, self.disp_max, self.disp_min)
                else:
                    disp_select = disp_select * self.disp_max
                    fake_vs = fake_vs * self.vs_max
                    reconstr_disp = reconstr_disp * self.disp_max

            mod_num = np.arange(0, 100, 33)
            period = np.array((0, 6, 9, 12, 15))
            period_tick = np.array((3, 6, 9, 12, 15))
            dep = np.arange(0, 99, 24)
            dep_tick = np.array((0, 12, 24, 36, 48))

            for j in range(c):
                if j == 0:
                    im = axs[j].imshow(disp_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(period)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    axs[j].set_ylabel('Period (s)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(period_tick, fontsize=16)
                    #cbar = plt.colorbar(im, orientation="horizontal", location='top', ax=axs[j])
                    #cbar.set_label('Velocity (km/s)', fontsize=12)
                elif j == 1:
                    im = axs[j].imshow(disp_select[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(period)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    #axs[j].set_ylabel('Period (s)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(period_tick, fontsize=16)
                    #cbar = fig.colorbar(im, ax=axs[:2], shrink=1.0, location='bottom')
                    #cbar.ax.tick_params(labelsize=20)
                    #cbar.set_label('Velocity (km/s)', fontsize=24)
                elif j == 2:
                    im = axs[j].imshow(fake_vs[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(dep)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    axs[j].set_ylabel('Depth (km)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(dep_tick, fontsize=16)
                    #plt.colorbar(im, ax=axs[j])
                elif j == 3:
                    im = axs[j].imshow(vs_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(dep)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    #axs[j].set_ylabel('Depth (km)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(dep_tick, fontsize=16)
                    #cbar = fig.colorbar(im, ax=axs[2:4], shrink=1.0, location='bottom')
                    #cbar.ax.tick_params(labelsize=20)
                    #cbar.set_label('$V_{s}$ (km/s)', fontsize=24)
                elif j == 4:
                    im = axs[j].imshow(reconstr_disp[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(period)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    axs[j].set_ylabel('Period (s)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(period_tick, fontsize=16)
                    #plt.colorbar(im, ax=axs[j])
                else:
                    im = axs[j].imshow(reconstr_disp[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    axs[j].set_xticks(mod_num)
                    axs[j].set_yticks(period)
                    axs[j].set_xlabel('Model number', fontsize=20)
                    #axs[j].set_ylabel('Period (s)', fontsize=20)
                    axs[j].set_xticklabels(mod_num+1, fontsize=16)
                    axs[j].set_yticklabels(period_tick, fontsize=16)
                    #cbar = fig.colorbar(im, ax=axs[4:6], shrink=1.0, location='bottom')
                    #cbar.ax.tick_params(labelsize=20)
                    #cbar.set_label('Velocity (km/s)', fontsize=24)
                axs[j].set_title(titles[j], fontsize=20)
                #axs[j].axis('off')
            fig.savefig(self.out_path + "predict/" + "V2Vs_winvgp_test")
            plt.close()

        else:
            titles = ['True Vs', 'Translated Vph', 'Translated Vgp', 'True Vph', 'True Vgp',
                      'Reconstructed Vs']
            disp_select, vs_select = self.data_loader.load_label_data(nselect=100, sample=True)

            if self.data_normalize:
                if self.balance_norm:
                    vs_select = balance_normalization(vs_select, self.vs_max, self.vs_min)
                else:
                    vs_select = vs_select / self.vs_max

            fig, axs = plt.subplots(1, c, figsize=(28, 14))

            fake_disp, reconstr_vs = self.sess.run(
                [self.testG, self.reconstrP],
                feed_dict={self.test_P: vs_select})

            if self.data_normalize:
                if self.balance_norm:
                    vs_select = balance_norm_reverse(vs_select, self.vs_max, self.vs_min)
                    fake_disp = balance_norm_reverse(fake_disp, self.disp_max, self.disp_min)
                    reconstr_vs = balance_norm_reverse(reconstr_vs, self.vs_max, self.vs_min)
                else:
                    vs_select = vs_select * self.vs_max
                    fake_disp = fake_disp * self.disp_max
                    reconstr_vs = reconstr_vs * self.vs_max

            for j in range(c):
                if j == 0:
                    im = axs[j].imshow(vs_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 1:
                    im = axs[j].imshow(fake_disp[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 2:
                    im = axs[j].imshow(fake_disp[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 3:
                    im = axs[j].imshow(disp_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 4:
                    im = axs[j].imshow(disp_select[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                else:
                    im = axs[j].imshow(reconstr_vs[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    plt.colorbar(im, ax=axs[j])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
            fig.savefig(self.out_path + "predict/" + "Vs2V_winvgp_test")
            plt.close()

    def rms_misfit(self, mode='Test'):
        """ Generate predicted full seismic vs model and compute the rms misfit """

        nbatch = self.Lbatch_rate // 2
        ntrace = self.nlabel
        bsize = self.nlabel // nbatch
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        Ldisp_data_orig, Lvs_data_orig, _, self.disp_max, self.vs_max, self.disp_min, self.vs_min, _, _ \
            = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.vs_max, self.disp_min, self.vs_min)

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed ...")

        if self.data_normalize:
            if self.balance_norm:
                Ldisp_data_orig = balance_normalization(Ldisp_data_orig, self.disp_max, self.disp_min)
            else:
                Ldisp_data_orig = Ldisp_data_orig / self.disp_max

        fake_vs = np.zeros((Ldisp_data_orig.shape[0], self.vs_dim, 1))

        for ibatch in range(nbatch + 1):
            print("Predicting ibatch %d/%d" % (ibatch, nbatch))
            if ibatch == nbatch:
                left = ibatch * bsize
                right = ntrace - 1
            else:
                left = ibatch * bsize
                right = (ibatch + 1) * bsize
            disp_batch = Ldisp_data_orig[left:right]

            fake_vs_batch, _ = self.sess.run(
                [self.testP, self.reconstrG],
                feed_dict={self.test_G: disp_batch})

            fake_vs[left:right, :, :] = fake_vs_batch

        if self.data_normalize:
            if self.balance_norm:
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
            else:
                fake_vs = fake_vs * self.vs_max

        loss = math.sqrt(np.sum((fake_vs[:,:,:] - Lvs_data_orig[:,:,:])**2)/fake_vs.shape[0]/fake_vs.shape[1])
        return loss

    def sample_vsprofile(self, epoch):
        r, c = 1, 5

        titles = ['Test Vph', 'Test Vgp', 'Translated', 'Reconstruct Vph', 'Reconstruct Vgp']
        fig, axs = plt.subplots(r, c, figsize=(28, 14))
        for i in range(r):
            
            disp_test = self.data_loader.load_test_data(nselect=100, sample=True, tdtype=self.test_dtype)
            
            if self.data_normalize:
                if self.balance_norm:
                    disp_test = balance_normalization(disp_test, self.disp_max, self.disp_min)
                else:
                    disp_test = disp_test / self.disp_max
            
            fake_vs, reconstr_disp = self.sess.run(
                    [self.testP, self.reconstrG],
                    feed_dict={self.test_G: disp_test})
            
            if self.data_normalize:
                if self.balance_norm:
                    disp_test = balance_norm_reverse(disp_test, self.disp_max, self.disp_min)
                    fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                    reconstr_disp = balance_norm_reverse(reconstr_disp, self.disp_max, self.disp_min)
                else:
                    disp_test = disp_test * self.disp_max
                    fake_vs = fake_vs * self.vs_max
                    reconstr_disp = reconstr_disp * self.disp_max

            for j in range(c):
                if j == 0:
                    im = axs[j].imshow(disp_test[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 1:
                    im = axs[j].imshow(disp_test[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 2:
                    im = axs[j].imshow(fake_vs[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 3:
                    im = axs[j].imshow(reconstr_disp[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                else:
                    im = axs[j].imshow(reconstr_disp[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
            fig.savefig(self.out_path+"Vsmodel_winvgp_e%d" % epoch)
            plt.close()
            
    def label_vsprofile(self, epoch):
        c = 6
        nselect = 100
        titles = ['Labeled Vph', 'Labeled Vgp', 'Translated Vs', 'Labeled Vs', 'Reconstructed Vph', 'Reconstructed Vgp']
        disp_select, vs_select = self.data_loader.load_label_data(nselect=nselect, sample=True)
        
        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_normalization(disp_select, self.disp_max, self.disp_min)
            else:
                disp_select = disp_select / self.disp_max
        
        fig, axs = plt.subplots(1, c, figsize=(28, 14))
        
        fake_vs, reconstr_disp = self.sess.run(
                [self.testP, self.reconstrG],
                feed_dict={self.test_G: disp_select})

        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_norm_reverse(disp_select, self.disp_max, self.disp_min)
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                reconstr_disp = balance_norm_reverse(reconstr_disp, self.disp_max, self.disp_min)
            else:
                disp_select = disp_select * self.disp_max
                fake_vs = fake_vs * self.vs_max
                reconstr_disp = reconstr_disp * self.disp_max

        loss = math.sqrt(np.sum((fake_vs[:,:,:] - vs_select[:,:,:])**2)/vs_select.shape[0]/vs_select.shape[1])

        for j in range(c):
            if j == 0:
                im = axs[j].imshow(disp_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.disp_min, self.disp_max)
                plt.colorbar(im, ax=axs[j])
            elif j == 1:
                im = axs[j].imshow(disp_select[:, :, 1].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.disp_min, self.disp_max)
                plt.colorbar(im, ax=axs[j])
            elif j == 2:
                im = axs[j].imshow(fake_vs[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.vs_min, self.vs_max)
                plt.colorbar(im, ax=axs[j])
            elif j == 3:
                im = axs[j].imshow(vs_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.vs_min, self.vs_max)
                plt.colorbar(im, ax=axs[j])
            elif j == 4:
                im = axs[j].imshow(reconstr_disp[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.disp_min, self.disp_max)
                plt.colorbar(im, ax=axs[j])
            else:
                im = axs[j].imshow(reconstr_disp[:, :, 1].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.disp_min, self.disp_max)
                plt.colorbar(im, ax=axs[j])
            axs[j].set_title(titles[j])
            axs[j].axis('off')
        fig.savefig(self.out_path+"Labeled_winvgp_e%d" % epoch)
        plt.close()

        return loss

    def line_vsprofile(self, epoch):
        c = 4

        # Load data from both training data and testing data
        disp_train, vs_train, pick_train = self.data_loader.load_label_data(nselect=200, sample=True,
                                                                           randpick=False, getpick=True)
        disp_test, pick_test = self.data_loader.load_test_data(nselect=1000, sample=True,
                                                                       randpick=False, getpick=True, tdtype=self.test_dtype)

        idex = [115, 60]
        disp_train = disp_train[idex]
        vs_train = vs_train[idex]
        pick_train = pick_train[idex]

        idex_test = [151, 600]
        disp_test = disp_test[idex_test]
        pick_test = pick_test[idex_test]

        if self.data_normalize:
            if self.balance_norm:
                disp_train = balance_normalization(disp_train, self.disp_max, self.disp_min)
                disp_test = balance_normalization(disp_test, self.disp_max, self.disp_min)
            else:
                disp_train = disp_train/self.disp_max
                disp_test = disp_test/self.disp_max

        fig, axs = plt.subplots(1, c, figsize=(18, 18))

        fake_vs_train, reconstr_disp_train = self.sess.run(
            [self.testP, self.reconstrG],
            feed_dict={self.test_G: disp_train})

        fake_vs_test, reconstr_disp_test = self.sess.run(
            [self.testP, self.reconstrG],
            feed_dict={self.test_G: disp_test})

        if self.data_normalize:
            if self.balance_norm:
                disp_train = balance_norm_reverse(disp_train, self.disp_max, self.disp_min)
                fake_vs_train = balance_norm_reverse(fake_vs_train, self.vs_max, self.vs_min)
                reconstr_disp_train = balance_norm_reverse(reconstr_disp_train, self.disp_max, self.disp_min)
                disp_test = balance_norm_reverse(disp_test, self.disp_max, self.disp_min)
                fake_vs_test = balance_norm_reverse(fake_vs_test, self.vs_max, self.vs_min)
                reconstr_disp_test = balance_norm_reverse(reconstr_disp_test, self.disp_max, self.disp_min)
            else:
                disp_train = disp_train * self.disp_max
                fake_vs_train = fake_vs_train * self.vs_max
                reconstr_disp_train = reconstr_disp_train * self.disp_max
                disp_test = disp_test * self.disp_max
                fake_vs_test = fake_vs_test * self.vs_max
                reconstr_disp_test = reconstr_disp_test * self.disp_max

        dz = 0.5
        dep = 0.0 + np.arange(self.vs_dim) * dz
        data = np.zeros((self.vs_dim, 6))
        for j in range(c):
            if j < 2:
                trace = vs_train[j, :, 0]
                pred_trace = fake_vs_train[j, :, 0]
                axs[j].plot(trace, dep, label='True Vs T%d' % (pick_train[j],))
                axs[j].plot(pred_trace, dep, label='Predict Vs T%d' % (pick_train[j],))
                axs[j].set(xlabel='Vs /km')
                axs[j].set(ylabel='Depth')
                axs[j].legend(loc='upper right')
                axs[j].label_outer()
                data[:, 2*j] = trace
                data[:, 2*j+1] = pred_trace
            else:
                k = j-2
                pred_trace = fake_vs_test[k, :, 0]
                axs[j].plot(pred_trace, dep, label='Predict Vs T%d' % (pick_test[k],))
                axs[j].set(xlabel='Vs /km')
                axs[j].set(ylabel='Depth')
                axs[j].legend(loc='upper right')
                axs[j].label_outer()
                data[:, j+2] = pred_trace
            axs[j].invert_yaxis()
        fig.savefig(self.out_path + "predict/" + "Sample_winvgp2d_e%d" % epoch)
        plt.close()
        np.savetxt(self.out_path+"predict/"+"WCGAN_train_%d_%d_test_%d_%d.txt" % (pick_train[0], pick_train[1], pick_test[0], pick_test[1]), data, '%3.4f')

if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    gan = CycleGAN(sess)
    mode = 'Train'
    if mode == 'Train':
        gan.train(epochs=1201, batch_size=5, sample_interval=25, startfrombeg=True)
    elif mode == 'Test':
        gan.test(direction='G2P', drawline=True)
        loss = gan.rms_misfit(mode=mode)
        print(loss)
    elif mode == 'Predict':
        gan.predict_vol(direction='G2P')
    else:
        print("Only training, testing and predicting modules are available")