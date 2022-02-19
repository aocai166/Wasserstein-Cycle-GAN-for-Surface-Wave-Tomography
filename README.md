# Wasserstein-Cycle-GAN-for-Surface-Wave-Tomography
Code and Benchmark Examples for a paper in JGR: Solid Earth

**Cai, A., Qiu, H., & Niu, F. (2022). Semi-supervised surface wave tomography with Wasserstein cycle-consistent GAN: Method and application to Southern California plate boundary region** <br />
Journal of Geophysical Research: Solid Earth, 127, e2021JB023598. <br />
https://doi.org/10.1029/2021JB023598

## Basic Notations <br />
The name of the deep learning method is **Wasserstein Cycle-GAN-GP (WCycle-GAN)**, a hybrid method of **Wasserstein GAN-GP** (*Arjovsky and Bottou, 2017; Arjovsky et al., 2017*) and **Cycle-GAN** (*Yi et al., 2017; Zhu et al., 2017*). **GP** stands for gradient penalty.

You can redistribute it and/or modify it under the terms of the GNU General Public License version 3.0. <br />
If you show inversion results in a paper or presentation please give a reference to the JGR paper

If you have questions when using the package, you can contact me at aocai166@gmail.com <br />

The code is writing in Python using the **TensorFlow** framework. Benchmark examples are given below. <br />

## Benckmark Examples
### (1) Setup your environment <br />
The Anaconda enviroment file (*environment.yml*) used in Cai et al. (2022) is provided. <br />
You can copy my environment using the yml file. <br />

An **tutorial** can be found at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

**Note**: due to the time of development of this code (2020), I used **TensorFlow 1.14**. It might not be compatable with the recent TensorFlow 2 and above versions.

### (2) Download the data (**Folder 'Data/'**) <br />
#### Labeled data <br />
Generated by first extracting 1-D *Vs* profiles from the Southern California Earthquake Center (SCEC) Community Velocity Model of *Shaw et al. (2015; CVMH)*. <br />
Then the theoretical Rayleigh wave dispersion curves are calculated using the *Herrmann (2013)* package.

There are three set of labeled data. <br />
- The **16480** labeled dispersion data (**Vs_region** & **disp_region**) <br />
- The **16480** labeled dispersion data + Position information (**Vs_region** & **disp_pos_region**) <br />
- The downsampled **1890** labeled dispersion data (**Vs_grid01** & **disp_gird01**) <br />

The dataset consists of text file for every grid cell. The name of the file contains the information of its latitude and longitude. For instance:

<pre> 1_32.600_239.800.txt  --> Latitude 32.6N; Longitude 120.2W. </pre>

For the data files, they include: <br />
<pre> Periods (3s-16s)    Phase Velocity (km/s)    Group Velocity (km/s)    Latitude    Longtitude </pre>
For the Vs profiles, they include: <br />
<pre> Depth (0km-49.0km)    Shear Wave Velocity (Vs; km/s) </pre>

#### Unlabeled data <br />
The unlabeled dispersion curves are 4076 observed Rayleigh wave dispersion curves extracted from *Qiu et al. (2019)*. They are stored as a 3-D matrix in *.npy* format for simplicity <br />

**Test_data_Qiu.npy** <br />
A (4076, 17, 5) matrix, containing the information of <br />
<pre> Longtitude    Latitude    Periods (3s-16s)    Phase Velocity (km/s)    Group Velocity (km/s) </pre>

**Test_data_Qiu_sigma.npy** <br />
A (4076, 17, 7) matrix, containing the information of <br />
<pre> Longtitude    Latitude    Periods (3s-16s)    Phase Velocity (km/s)    Group Velocity (km/s)    Uncertainty of Phase Velocity (km/s)    Uncertainty of Group Velocity (km/s)</pre>

### (3) Download the source code (**Folder 'Src/'**) <br />
The source code folder contains deep learning Vs inversion of several methods presented in *Cai et al., (2020)*, including **Convolutional Neural Networks (CNN)**, **Cycle-GAN or LS-Cycle-GAN**, **Wcycle-GAN**, **Wcycle-GAN+Position**.

The instruction of the codes are very similar. Here the Wcycle-GAN example is illustrated. <br />

The data used in the training is controled in the (***Vs_inv_data_loader.py***)
```
self.file_disp_path = self.file_train_path + 'disp_region/'
self.file_vs_path = self.file_train_path + 'Vs_region/'
self.test_disp_path = test_disp_path
self.out_path = out_path
#self.tdname = 'test_data_Qiu.npy'
self.tdname = 'test_data_Qiu_sigma.npy'
```
The labeled data '**disp_region/**' and '**Vs_region/**' and unlabeled data '**test_data_Qiu_sigma.npy**' are explained in the previous section.

The main body of the Wcycle-GAN based Vs inversion is at (***Wcyclegan-gp-tf_Vs_inv_1D.py***)
First check some basic parameters and file settings (the values are ready for the benckmark examples)
```
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
```
Here the dispersion data is of dimension *17x2*, and Vs model is of dimension *99x1*. **Lsize** and **Usize** are the batch sizes of labeled and unlabeled data used in training, respectively. <br />

Change the ***self.file_train_path*** to the path of where you have put the data folder. <br />
Change ***self.out_path*** to where you want to have figure outputs and final models. <br />
You don't necessarily need to change ***self.test_disp_path***, but just in case, make it the same as the ***file_train_path***. <br />
```
# Path to the folder of labeled data (e.g., disp_region/)
self.file_train_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Train_dat/'
# Path to the folder of labeled data (not used as the test data is a numpy file)
self.test_disp_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/linear_test/'
# Path to the directory where you want to store the results
self.out_path = 'D:/PycharmProjects/GPUpy/Qiu_data/CVMHTrainingDataset/Final_results/output_region_chi/'
# type of test data, vary between txt and npy
self.test_dtype = 'npy'
self.save_freq = 25
```
***self.save_freq*** is the frequency that you would like to save the network parameters (per 25 epochs)

Near the end lines of the code, you can find the code blocks
```
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
```
There are several modules, including <br />


To control trian/test/apply on unlabeled data, look at the bottom of "Wcyclegan-gp-tf_Qiu_1D.py", you will see "Train", "Test" and "Predict" modes are available. Simply change the mode value will work. But when you want to use "Test" and "Predict" module, please first generate a folder named "predict" in your output directory.

![Figure7](https://user-images.githubusercontent.com/35436104/154765285-227c78f2-667c-4b53-a232-7c6fb84e2e75.JPG)

## References
*Arjovsky, M., & Bottou, L. (2017). Towards Principled Methods for Training Generative Adversarial Networks. 5th International Conference on Learning Representations. Retrieved from https://arxiv.org/abs/1701.04862v1*

*Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. Retrieved from http://arxiv.org/abs/1701.07875*

*Herrmann, R. B. (2013). Computer programs in seismology: An evolving tool for instruction and research. Seismological Research Letters, 84(6), 1081–1088. https://doi.org/10.1785/0220110096*

*Qiu, H., Lin, F. C., & Ben-Zion, Y. (2019). Eikonal Tomography of the Southern California Plate Boundary Region. Journal of Geophysical Research: Solid Earth, 124(9), 9755–9779. https://doi.org/10.1029/2019JB017806*

*Shaw, J. H., Plesch, A., Tape, C., Suess, M. P., Jordan, T. H., Ely, G., et al. (2015). Unified Structural Representation of the southern California crust and upper mantle. Earth and Planetary Science Letters, 415, 1–15. https://doi.org/10.1016/j.epsl.2015.01.016*

*Yi, Z., Zhang, H., Tan, P., & Gong, M. (2017). DualGAN: Unsupervised Dual Learning for Image-to-Image Translation. Proceedings of the IEEE International Conference on Computer Vision, 2017-October, 2868–2876. https://doi.org/10.1109/ICCV.2017.310*

*Zhu, J.-Y., Park, T., Isola, P., Efros, A. A., & Research, B. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks Monet Photos. Retrieved from https://github.com/junyanz/CycleGAN.*
