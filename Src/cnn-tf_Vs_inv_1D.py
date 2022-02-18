#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 02/18/2022

@author: aocai (aocai166@gmail.com) Rice University
"""
# Convolutional Neural Network (CNN) with Least-sqaures loss
# using tensorflow for deriving Vs model from dispersion curve
# Labeled data: Extracted from Community velocity model of Shaw et al. (CVMH) 16480 dispersion curves
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
from Vs_inv_data_loader_v1 import DataLoader


def balance_normalization(array, gvalmax, gvalmin):
    return 2.0 * (array-gvalmin)/(gvalmax-gvalmin) - 1.0


def balance_norm_reverse(array, gvalmax, gvalmin):
    return gvalmin + (array + 1.0) * (gvalmax-gvalmin)/2.0


class ConvNet():
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
        self.var_rms = 0.07123354953152737  # RMS averaged uncertainty value of the unlabeled data

        # Path to the folder of labeled data (e.g., disp_region/)
        self.file_train_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Train_dat/'
        # Path to the folder of labeled data (not used as the test data is a numpy file)
        self.test_disp_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/linear_test/'
        # Path to the directory where you want to store the results
        self.out_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Final_results/output_cnn/'
        self.test_dtype = 'npy'                                            # type of test data, vary between txt and npy
        self.save_freq = 20
        
        # Configure data Loader
        self.data_loader = DataLoader(file_train_path=self.file_train_path, test_disp_path=self.test_disp_path,
                                      out_path=self.out_path, nbatch_label=self.Lbatch_rate,
                                      nbatch_ulabel=self.Ubatch_rate,
                                      ntrain=self.nlabel, ntest=self.ulabel,
                                      vs_sample=self.vs_dim, disp_sample=self.disp_dim)

        # Number of filters in the first layer of ConvNet
        self.gf = 32

        # Initialize cycle-gan structures
        if self.gen_archi == 'resnet':
            self.cnnG2P = generator_resnet_G2P(input_shape=self.disp_shape, gf=self.gf, name='generatorG2P')
        else:
            self.cnnG2P = generator_dcnn_G2P(input_shape=self.disp_shape, gf=self.gf, name='generatorG2P')
        print(self.cnnG2P.summary())

        self.criterionCNN = mse_criterion

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        # Labeled impedance data
        self.real_Limp = tf.placeholder(tf.float32, [None, self.vs_dim, self.vs_channels], name="real_vs_label")
        # Labeled image data
        self.real_Limg = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="real_disp_label")

        # When we train the generator, we freeze the discriminator
        self.cnnG2P.trainable = True

        self.fake_Limp = self.cnnG2P(self.real_Limg)
        self.cnn_loss = self.criterionCNN(self.real_Limp, self.fake_Limp)
        self.cnn_loss_sum = tf.summary.scalar("cnn_loss", self.cnn_loss)

        self.test_G = tf.placeholder(tf.float32, [None, self.disp_dim, self.disp_channels], name="test_disp")
        self.testP = self.cnnG2P(self.test_G)

        t_vars = tf.trainable_variables()
        self.cnn_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, epochs, batch_size=10, sample_interval=1):
        """ Train Wasserstein Cycle GAN """

        self.cnn_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.cnn_loss, var_list=self.cnn_vars)
        #self.cnn_optim = tf.train.RMSPropOptimizer(self.lr, beta1=0.5).minimize(self.cnn_loss, var_list=self.cnn_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.out_path+"logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        cnn_loss_val = []

        Ldisp_data_orig, Lvs_data_orig, self.disp_max, self.vs_max, self.disp_min, self.vs_min,\
            self.file_name_train, self.file_name_test = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.disp_min, self.vs_max, self.vs_min)

        # Normalize the data
        if self.data_normalize:
            if self.balance_norm:
                Ldisp_data_orig = balance_normalization(Ldisp_data_orig, self.disp_max, self.disp_min)
                Lvs_data_orig = balance_normalization(Lvs_data_orig, self.vs_max, self.vs_min)
            else:
                Ldisp_data_orig = Ldisp_data_orig/self.disp_max
                Lvs_data_orig = Lvs_data_orig/self.vs_max

        print(np.max(Ldisp_data_orig), np.min(Ldisp_data_orig), np.max(Lvs_data_orig), np.min(Lvs_data_orig))

        for epoch in range(epochs):
            if epoch == 0:
                self.sample_vsprofile(epoch)
                loss_val = self.label_vsprofile(epoch)

            #idx = np.random.permutation(self.ulabel)
            #idx = np.random.randint(self.ulabel, size=self.ulabel)

            nbatch = max(self.Lbatch_rate, self.Ubatch_rate)

            for ibatch in range(nbatch):
                Lkbatch = ibatch % self.Lbatch_rate
                if Lkbatch == 0:
                    Lidx = np.random.permutation(self.nlabel)

                # Batch the labeled data and permute
                Lleft = Lkbatch * self.Lsize
                Lright = (Lkbatch+1) * self.Lsize
                Lidx_batch = Lidx[Lleft: Lright]
                Ldisp_data = Ldisp_data_orig[Lidx_batch]
                Lvs_data = Lvs_data_orig[Lidx_batch]

                # Update G network and record fake outputs
                _, _, summary_str = self.sess.run(
                        [self.cnn_loss, self.cnn_optim, self.cnn_loss_sum], feed_dict={self.real_Limg: Ldisp_data,
                                                                        self.real_Limp: Lvs_data})

                #loss_val = math.sqrt(loss_val)
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d/%2d] Lbatch: [%4d/%4d] Loss=%4.4f time: %4.4f" % (
                        epoch, epochs, Lkbatch, self.Lbatch_rate, loss_val, time.time() - start_time)))

            if epoch % self.save_freq == 0:
                self.save(epoch)

            if epoch % sample_interval == 0:
                self.sample_vsprofile(epoch)
                loss_val = self.label_vsprofile(epoch)
                cnn_loss_val.append((epoch, loss_val))
                np.savetxt(self.out_path + 'Loss_func.txt', cnn_loss_val)

    def save(self, step):
        model_name = "ConvNet.model"
        model_dir = self.out_path
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.saver.save(self.sess,
                        os.path.join(model_dir, model_name),
                        global_step=step)

    def load(self, specify=False, imodel=1000):
        """ Load trained Wcycle-GAN model. If want to load specified model, put specify=True and give epoch number"""
        print("[*] Reading checkpoint ...")

        model_dir = self.out_path
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            if specify:
                tmp_name = os.path.basename(ckpt.model_checkpoint_path)
                tmp_name = tmp_name.split('-')
                ckpt_name = tmp_name[0] + '-' + str(imodel)
            else:
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
        _, _, self.disp_max, self.vs_max, self.disp_min, self.vs_min, self.file_name_train, self.file_name_test \
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

        for ibatch in range(nbatch + 1):
            print("Predicting ibatch %d/%d" % (ibatch, nbatch))
            if ibatch == nbatch:
                left = ibatch * bsize
                right = ntrace - 1
            else:
                left = ibatch * bsize
                right = (ibatch + 1) * bsize
            disp_batch = disp_full[left:right]

            fake_vs_batch = self.sess.run(
                self.testP, feed_dict={self.test_G: disp_batch})

            fake_vs[left:right, :, :] = fake_vs_batch

        if self.data_normalize:
            if self.balance_norm:
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
            else:
                fake_vs = fake_vs * self.vs_max

        np.save(self.out_path + 'predict/' + 'Predict_vs.npy', fake_vs)

    def test(self, direction='G2P', drawline=False):
        """ Test ConvNet """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        _, _, self.disp_max, self.vs_max, self.disp_min, self.vs_min, self.file_name_train, self.file_name_test\
            = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.vs_max, self.disp_min, self.vs_min)

        if self.load():
            print("[*] Load SUCCESS")
        else:
            print("[!] Load failed ...")

        if drawline:
            self.line_vsprofile(0)

        c = 4

        titles = ['Labeled Vph', 'Labeled Vgp', 'Translated Vs', 'Labeled Vs']
        disp_select, vs_select = self.data_loader.load_label_data(nselect=100, sample=True)

        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_normalization(disp_select, self.disp_max, self.disp_min)
            else:
                disp_select = disp_select / self.disp_max

        fig, axs = plt.subplots(1, c, figsize=(28, 14))

        fake_vs = self.sess.run(self.testP, feed_dict={self.test_G: disp_select})

        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_norm_reverse(disp_select, self.disp_max, self.disp_min)
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
            else:
                disp_select = disp_select * self.disp_max
                fake_vs = fake_vs * self.vs_max

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
            else:
                im = axs[j].imshow(vs_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.vs_min, self.vs_max)
                plt.colorbar(im, ax=axs[j])
            axs[j].set_title(titles[j])
            axs[j].axis('off')
        fig.savefig(self.out_path + "predict/" + "V2Vs_winvgp_test")
        plt.close()

    def rms_misfit(self, mode='Test'):
        """ Generate predicted full seismic vs model and compute the rms misfit """

        nbatch = self.Lbatch_rate // 2
        ntrace = self.nlabel
        bsize = self.nlabel // nbatch
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        Ldisp_data_orig, Lvs_data_orig, self.disp_max, self.vs_max, self.disp_min, self.vs_min, _, _ \
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

            fake_vs_batch = self.sess.run(self.testP, feed_dict={self.test_G: disp_batch})

            fake_vs[left:right, :, :] = fake_vs_batch

        if self.data_normalize:
            if self.balance_norm:
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
            else:
                fake_vs = fake_vs * self.vs_max

        loss = math.sqrt(np.sum((fake_vs[:,:,:] - Lvs_data_orig[:,:,:])**2)/fake_vs.shape[0]/fake_vs.shape[1])
        return loss

    def sample_vsprofile(self, epoch):
        r, c = 1, 3

        titles = ['Test Vph', 'Test Vgp', 'Translated']
        fig, axs = plt.subplots(r, c, figsize=(28, 14))
        for i in range(r):
            
            disp_test = self.data_loader.load_test_data(nselect=100, sample=True, tdtype=self.test_dtype)
            
            if self.data_normalize:
                if self.balance_norm:
                    disp_test = balance_normalization(disp_test, self.disp_max, self.disp_min)
                else:
                    disp_test = disp_test / self.disp_max
            
            fake_vs = self.sess.run(self.testP, feed_dict={self.test_G: disp_test})
            
            if self.data_normalize:
                if self.balance_norm:
                    disp_test = balance_norm_reverse(disp_test, self.disp_max, self.disp_min)
                    fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                else:
                    disp_test = disp_test * self.disp_max
                    fake_vs = fake_vs * self.vs_max

            for j in range(c):
                if j == 0:
                    im = axs[j].imshow(disp_test[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                elif j == 1:
                    im = axs[j].imshow(disp_test[:, :, 1].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.disp_min, self.disp_max)
                    plt.colorbar(im, ax=axs[j])
                else:
                    im = axs[j].imshow(fake_vs[:, :, 0].T, cmap='rainbow', aspect='auto')
                    im.set_clim(self.vs_min, self.vs_max)
                    plt.colorbar(im, ax=axs[j])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
            fig.savefig(self.out_path+"Vsmodel_winvgp_e%d" % epoch)
            plt.close()
            
    def label_vsprofile(self, epoch):
        c = 4
        titles = ['Labeled Vph', 'Labeled Vgp', 'Translated Vs', 'Labeled Vs']
        disp_select, vs_select = self.data_loader.load_label_data(nselect=100, sample=True)
        
        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_normalization(disp_select, self.disp_max, self.disp_min)
            else:
                disp_select = disp_select / self.disp_max
        
        fig, axs = plt.subplots(1, c, figsize=(28, 14))
        
        fake_vs = self.sess.run(self.testP, feed_dict={self.test_G: disp_select})

        if self.data_normalize:
            if self.balance_norm:
                disp_select = balance_norm_reverse(disp_select, self.disp_max, self.disp_min)
                fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
            else:
                disp_select = disp_select * self.disp_max
                fake_vs = fake_vs * self.vs_max

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
            else:
                im = axs[j].imshow(vs_select[:, :, 0].T, cmap='rainbow', aspect='auto')
                im.set_clim(self.vs_min, self.vs_max)
                plt.colorbar(im, ax=axs[j])
            axs[j].set_title(titles[j])
            axs[j].axis('off')
        fig.savefig(self.out_path+"Labeled_winvgp_e%d" % epoch)
        plt.close()

        return loss

    def line_vsprofile(self, epoch):
        c = 4

        # Load data from both training data and testing data
        disp_train, vs_train, pick_train = self.data_loader.load_label_data(nselect=2, sample=True,
                                                                           randpick=True, getpick=True)
        disp_test, pick_test = self.data_loader.load_test_data(nselect=2, sample=True,
                                                                       randpick=True, getpick=True, tdtype=self.test_dtype)

        if self.data_normalize:
            if self.balance_norm:
                disp_train = balance_normalization(disp_train, self.disp_max, self.disp_min)
                disp_test = balance_normalization(disp_test, self.disp_max, self.disp_min)
            else:
                disp_train = disp_train/self.disp_max
                disp_test = disp_test/self.disp_max

        fig, axs = plt.subplots(1, c, figsize=(18, 18))

        fake_vs_train = self.sess.run(self.testP, feed_dict={self.test_G: disp_train})

        fake_vs_test = self.sess.run(self.testP, feed_dict={self.test_G: disp_test})

        if self.data_normalize:
            if self.balance_norm:
                disp_train = balance_norm_reverse(disp_train, self.disp_max, self.disp_min)
                fake_vs_train = balance_norm_reverse(fake_vs_train, self.vs_max, self.vs_min)
                disp_test = balance_norm_reverse(disp_test, self.disp_max, self.disp_min)
                fake_vs_test = balance_norm_reverse(fake_vs_test, self.vs_max, self.vs_min)
            else:
                disp_train = disp_train * self.disp_max
                fake_vs_train = fake_vs_train * self.vs_max
                disp_test = disp_test * self.disp_max
                fake_vs_test = fake_vs_test * self.vs_max

        dz = 0.5
        dep = 0.0 + np.arange(self.vs_dim) * dz
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
            else:
                k = j-2
                pred_trace = fake_vs_test[k, :, 0]
                axs[j].plot(pred_trace, dep, label='Predict Vs T%d' % (pick_test[k],))
                axs[j].set(xlabel='Vs /km')
                axs[j].set(ylabel='Depth')
                axs[j].legend(loc='upper right')
                axs[j].label_outer()
            axs[j].invert_yaxis()
        fig.savefig(self.out_path + "predict/" + "Sample_winvgp2d_e%d" % epoch)
        plt.close()

    def model_uncertainty_est(self, type='PerturbNets', noisetype='Gaussian',
                              epoch_start=1000, epoch_jump=1, njumps=1, nperturb=1):
        """ Estimate the model uncertainty of the final Vs prediction by perturb Data or Nets
        Starting epoch is epoch_start. The number of epochs jump is epoch_jump and number of model to test is njump"""

        nbatch = self.Ubatch_rate // 2
        ntrace = self.ulabel
        bsize = self.ulabel // nbatch
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        Ldisp_data_orig, Lvs_data_orig, self.disp_max, self.vs_max, self.disp_min, self.vs_min, \
        self.file_name_train, self.file_name_test = self.data_loader.load_data(fulldata=True, tdtype=self.test_dtype)
        print(self.disp_max, self.disp_min, self.vs_max, self.vs_min)


        if type == 'PerturbNets':
            output_dir = self.out_path + 'predict/' + 'PertNets/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                print("Directory", output_dir, "Created")
            else:
                print("Directory", output_dir, "Already exists")

            fake_vs_full = np.zeros((ntrace, self.vs_dim, njumps))

            epoch_curr = epoch_start

            for ijump in range(njumps):
                print("Predicting ijump %d/%d" % (ijump, njumps))

                epoch_curr = epoch_start + ijump*epoch_jump

                if self.load(specify=True, imodel=epoch_curr):
                    print(" [*] Load SUCCESS")
                else:
                    print(" [!] Load failed ...")
                    break

                test_full = self.data_loader.load_full_test_data()
                disp_full = np.zeros((test_full.shape[0], test_full.shape[1], 2))
                disp_full[:,:,0:2] = test_full[:, :, 3:5]

                if self.data_normalize:
                    if self.balance_norm:
                        disp_full[:, :, 0:2] = balance_normalization(disp_full[:, :, 0:2], self.disp_max, self.disp_min)
                    else:
                        disp_full = disp_full / self.disp_max

                fake_vs = np.zeros((ntrace, self.vs_dim, self.vs_channels))

                for ibatch in range(nbatch + 1):
                    if ibatch == nbatch:
                        left = ibatch * bsize
                        right = ntrace - 1
                    else:
                        left = ibatch * bsize
                        right = (ibatch + 1) * bsize
                    disp_batch = disp_full[left:right]

                    fake_vs_batch = self.sess.run(self.testP,
                        feed_dict={self.test_G: disp_batch})

                    fake_vs[left:right, :, :] = fake_vs_batch

                if self.data_normalize:
                    if self.balance_norm:
                        fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                    else:
                        fake_vs = fake_vs * self.vs_max

                fake_vs_full[:, :, ijump] = fake_vs[:, :, 0]

            vs_name = 'Predict_vs_cnn_epoch_%d_%d.npy' % (epoch_start, epoch_curr)
            np.save(output_dir + vs_name, fake_vs_full)

        elif type == 'PerturbData':

            if self.load():
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed ...")
                return

            output_dir = self.out_path + 'predict/' + 'PertData/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                print("Directory", output_dir, "Created")
            else:
                print("Directory", output_dir, "Already exists")

            fake_vs_full = np.zeros((ntrace, self.vs_dim, nperturb))

            test_full = self.data_loader.load_full_test_data()
            disp_orig = np.zeros((test_full.shape[0], test_full.shape[1], 2))
            disp_orig[:, :, 0:2] = test_full[:, :, 3:5]

            disp_full = np.zeros((test_full.shape[0], test_full.shape[1], 2))
            var_full = test_full[:, :, 5:7]

            for iperturb in range(nperturb):
                # Generate random noise value with different seeds
                print("Predicting iperturb %d/%d" % (iperturb, nperturb))
                if noisetype == 'Gaussian':
                    noise = np.random.normal(loc=0, scale=1.0, size=(ntrace, disp_orig.shape[1], 2))
                elif noisetype == 'Uniform':
                    noise = np.random.uniform(low=-1.0, high=1.0, size=(ntrace, disp_orig.shape[1], 2))
                else:
                    print("Incorrect noise type, must select between 'Gaussian' and 'Uniform'")
                    return

                disp_full[:, :, 0:2] = disp_orig[:, :, 0:2] + self.var_rms * np.multiply(var_full, noise)

                if self.data_normalize:
                    if self.balance_norm:
                        disp_full[:, :, 0:2] = balance_normalization(disp_full[:, :, 0:2], self.disp_max, self.disp_min)
                    else:
                        disp_full = disp_full / self.disp_max

                fake_vs = np.zeros((ntrace, self.vs_dim, self.vs_channels))

                for ibatch in range(nbatch + 1):
                    if ibatch == nbatch:
                        left = ibatch * bsize
                        right = ntrace - 1
                    else:
                        left = ibatch * bsize
                        right = (ibatch + 1) * bsize
                    disp_batch = disp_full[left:right]

                    fake_vs_batch = self.sess.run(self.testP,
                        feed_dict={self.test_G: disp_batch})

                    fake_vs[left:right, :, :] = fake_vs_batch

                if self.data_normalize:
                    if self.balance_norm:
                        fake_vs = balance_norm_reverse(fake_vs, self.vs_max, self.vs_min)
                    else:
                        fake_vs = fake_vs * self.vs_max

                fake_vs_full[:, :, iperturb] = fake_vs[:, :, 0]

            vs_name = 'Predict_vs_cnn_perturbs_%d.npy' % (nperturb)
            np.save(output_dir + vs_name, fake_vs_full)

        else:
            print('Unsupport perturbation option! Please select between "PerturbNets" and "PerturbData"')
            return


if __name__ == "__main__":
    sess = tf.compat.v1.Session()
    cnn = ConvNet(sess)
    mode = 'Mod_uncer'
    if mode == 'Train':
        cnn.train(epochs=201, batch_size=5, sample_interval=20)
    elif mode == 'Test':
        cnn.test(direction='G2P', drawline=True)
        loss = cnn.rms_misfit(mode=mode)
        print(loss)
    elif mode == 'Predict':
        cnn.predict_vol(direction='G2P')
    elif mode == 'Mod_uncer':
        # cnn.model_uncertainty_est(type='PerturbNets', epoch_start=0, epoch_jump=10, njumps=11)
        cnn.model_uncertainty_est(type='PerturbData', nperturb=100, noisetype='Uniform')
    else:
        print("Only training, testing and predicting modules are available")