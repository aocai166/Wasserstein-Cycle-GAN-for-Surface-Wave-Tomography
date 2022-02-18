#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 02/18/2022

@author: aocai (aocai166@gmail.com) Rice University
"""

# Data loader for SC CVMH model/data and Hongrui Qiu's data
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, file_train_path, test_disp_path, out_path,
                 nbatch_label=100, nbatch_ulabel=100, ntrain=22580, ntest=2432, vs_sample=99, disp_sample=17):

        self.vs_sample = vs_sample
        self.disp_sample = disp_sample
        self.disp_max = 10
        self.vs_max = 10
        self.disp_min = 1
        self.vs_min = 1
        self.ntrain = ntrain
        self.ntest = ntest
        self.Lidx = 0
        self.Uidx = 0
        self.Lnbatch = nbatch_label
        self.Unbatch = nbatch_ulabel
        self.Lsize = self.ntrain // self.Lnbatch
        self.Usize = self.ntest // self.Unbatch

        self.file_train_path = file_train_path
        self.file_disp_path = self.file_train_path + 'disp_region/'
        self.file_vs_path = self.file_train_path + 'Vs_region/'
        self.test_disp_path = test_disp_path
        self.out_path = out_path
        self.file_name_train = self.load_train_dataname()
        self.file_name_test = []
#        self.tdname = 'test_data_Qiu.npy'
        self.tdname = 'test_data_Qiu_sigma.npy'
        tmp_vol = np.load(self.file_train_path + self.tdname)
        self.tdnpy = tmp_vol[:, :, 3:5]

    def load_train_dataname(self):
#        file = open(self.file_train_path+'file_train_v0.txt', 'r')
#        lst = file.readlines()
#        file.close()
#        out = lst[0].split(',')
        out = os.listdir(self.file_vs_path)
        print(len(out))
        if len(out) != self.ntrain:
            print('Load data size not matched with input data size')
            sys.exit(0)

        return out

    def load_data(self, fulldata=True, tdtype='npy'):

        Ldisp_data = np.zeros((self.ntrain, self.disp_sample, 2))
        Lvs_data = np.zeros((self.ntrain, self.vs_sample, 1))

        itrain = 0
        for file in self.file_name_train:
            tmp_disp = np.loadtxt(self.file_disp_path+file)
            Ldisp_data[itrain, :, :] = tmp_disp[:, 1:3]
            tmp_vs = np.loadtxt(self.file_vs_path+file)
            Lvs_data[itrain, :, 0] = tmp_vs[:, 1]
            itrain += 1

        train_disp_max = np.max(Ldisp_data)
        self.disp_max = train_disp_max
        train_disp_min = np.min(Ldisp_data)
        self.disp_min = train_disp_min
        self.vs_max = np.max(Lvs_data)
        self.vs_min = np.min(Lvs_data)

        print("Data loading finished")

        if fulldata:
            return Ldisp_data, Lvs_data, self.disp_max, self.vs_max, \
                   self.disp_min, self.vs_min, self.file_name_train, self.file_name_test
        else:
            return self.disp_max, self.vs_max, self.disp_min, self.vs_min,\
                   self.file_name_train, self.file_name_test

    def load_test_data(self, nselect=100, sample=True, seed=None, randpick=False, getpick=False, tdtype='npy'):
        select = []

        if sample:
            if randpick:
                np.random.seed(seed)
                select = np.random.randint(self.ntest, size=nselect)
            else:
                select = np.linspace(1, self.ntest-10, num=nselect)

            disp_test = np.zeros((nselect, self.disp_sample, 2))

            if tdtype == 'npy':
                itest = 0
                for iselect in select:
                    disp_test[itest, :, :] = self.tdnpy[int(iselect), :, :]
                    itest += 1
            else:
                itest = 0
                for iselect in select:
                    file = self.file_name_test[int(iselect)]
                    tmp_disp = np.loadtxt(self.test_disp_path+file)
                    disp_test[itest, :, :] = tmp_disp[:, 1:3]
                    itest += 1
        else:

            if tdtype == 'npy':
                disp_test = self.tdnpy
            else:
                disp_test = np.zeros((self.ntest, self.disp_sample, 2))

                itest = 0
                for file in self.file_name_test:
                    tmp_disp = np.loadtxt(self.test_disp_path+file)
                    disp_test[itest, :, :] = tmp_disp[:, 1:3]
                    itest += 1

        if getpick:
            return disp_test, select
        else:
            return disp_test

    def load_full_test_data(self, tdtype='npy'):
        if tdtype == 'npy':
            test_vol = np.load(self.file_train_path + self.tdname)
            return test_vol
        else:
            print('Full test data must be npy for now')
            sys.exit(-1)

    def load_label_data(self, nselect=100, sample=True, seed=None, randpick=False, getpick=False):
        select = []

        if sample:
            if randpick:
                np.random.seed(seed)
                select = np.random.randint(self.ntrain, size=nselect)
            else:
                select = np.linspace(1, self.ntrain-10, num=nselect)

            disp_select = np.zeros((nselect, self.disp_sample, 2))
            vs_select = np.zeros((nselect, self.vs_sample, 1))

            ipick = 0
            for iselect in select:
                disp_name = self.file_name_train[int(iselect)]
                vs_name = self.file_name_train[int(iselect)]

                tmp_disp = np.loadtxt(self.file_disp_path + disp_name)
                disp_select[ipick, :, :] = tmp_disp[:, 1:3]

                tmp_vs = np.loadtxt(self.file_vs_path + vs_name)
                vs_select[ipick, :, 0] = tmp_vs[:, 1]

                ipick += 1
        else:
            disp_select = np.zeros((self.ntrain, self.disp_sample, 2))
            vs_select = np.zeros((self.ntrain, self.disp_sample, 1))

            ipick = 0
            for file in self.file_name_train:
                tmp_disp = np.loadtxt(self.file_disp_path + file)
                disp_select[ipick, :, :] = tmp_disp[:, 1:3]

                tmp_vs = np.loadtxt(self.file_vs_path + file)
                vs_select[ipick, :, :] = tmp_vs[:, 1]

                ipick += 1

        if getpick:
            return disp_select, vs_select, select
        else:
            return disp_select, vs_select
