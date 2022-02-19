# Wasserstein-Cycle-GAN-for-Surface-Wave-Tomography
Code and Benchmark Examples for a paper in JGR: Solid Earth

**Cai, A., Qiu, H., & Niu, F. (2022). Semi-supervised surface wave tomography with Wasserstein cycle-consistent GAN: Method and application to Southern California plate boundary region** <br />
Journal of Geophysical Research: Solid Earth, 127, e2021JB023598. <br />
https://doi.org/10.1029/2021JB023598

## Basic Notations <br />
You can redistribute it and/or modify it under the terms of the GNU General Public License version 3.0. <br />
If you show inversion results in a paper or presentation please give a reference to the JGR paper

If you have questions when using the package, you can contact me at aocai166@gmail.com <br />

The code is writing in Python using the TensorFlow framework. A benchmark example is given below. <br />

## Benckmark Example
#### (1) Setup your environment <br />
The Anaconda enviroment file (environment.yml) used in Cai et al. (2022) is provided. You can copy my environment using the yml file. <br />
An tutorial can be found here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html <br />
Note: due to the time of development of this code (2020), I used TensorFlow 1.14. It might not be compatable with the recent TensorFlow 2 and above versions.

#### (2) Download the data and the codes <br />
All the labeled data from CVMH model and unlabeled dispersion curves from Qiu et al. (2019) are prepared in the folder 'Data/'.

To play around with this, please put the code and data to whatever folder you would like to in you computer/server. Look at "Wcyclegan-gp-tf_Vs_inv_1D.py", all the parameters are settled down. What you need to change is the "self.file_train_path" to the path of where you have put the data folder, change "self.out_path" to where you want to have figure outputs and final models. You don't necessarily need to change "self.test_disp_path", but just in case, make it the same as the file_train_path.

Then you can run this file and the training will start, you need to build up the environment to run this code, an easy way to know what to install is looking at all the "import ...", then you know what packages are needed, they can be easily installed in the anaconda environment. What's important is that please use the tensorflow version 1.14.0. The current code is not compatible with tensorflow version higher than 2.0.

To control trian/test/apply on unlabeled data, look at the bottom of "Wcyclegan-gp-tf_Qiu_1D.py", you will see "Train", "Test" and "Predict" modes are available. Simply change the mode value will work. But when you want to use "Test" and "Predict" module, please first generate a folder named "predict" in your output directory.

![Figure7](https://user-images.githubusercontent.com/35436104/154765285-227c78f2-667c-4b53-a232-7c6fb84e2e75.JPG)
