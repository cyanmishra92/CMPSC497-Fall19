# CMPSC497: Hands-on Deep Learning
## Term - Fall19, Pennsylvania State University
Projects and Example for cmpsc497-fall19 term
The course focuses on teachig the students a practical approach to build deep neural networks and deploy them using full stack developement setup. It covers from developing tensorflow models to train and build a network to accelerating the DNNs using sepcilized hardware e.g. FPGAs. A special focus will be on OpenCL based FPGA accelerator develpoment and letting students accelerate their DNNs using an FPGA. 
## Installing Tensorflow and Keras
1. ssh to one of the machines in W204 lab using the MobaXterm or any other ssh clients. This is only required if you are tryin to do it from yor laptop. If you are physically present in the lab, you can skip this step 
(This requires using cse vpn. For more information please visit https://ais.its.psu.edu/services/vpn/
Your vpn address will be vpn.cse.psu.edu. If you are not familiar with the VPN please see us in the office hours.)
    
    $ssh -XY <username>@cse-p204instxx.cse.psu.edu
For example: $ssh -XY cxm2114@cse-p204inst04.cse.psu.edu
This will ask you to authenticate for logging in. You can use your CSE password and 2FA to log in. 

2. Download the anaconda installtion script fromanaconda site (https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh)
        
        $wget "https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh"

3. Run the installtion script and follow the steps. This step should complete seamlesly. Please see us in the office hours if you are not able to do this step. 
    
        $bash Anaconda2-2019.10-Linux-x86_64.sh

4. After the installation the installer will prompt you to restart the terminal. Please open a new terminal (or log in again) to ensure the anaconda installtion is complte. 

5. Installing tensorflow environment: 
    
        $conda create -n tensorflow_env tensorflow
        
This will install tensorflow with all its dependencies. Please follow the installation steps. After installtion this will prompt you to restart the terminal. Please do the same for the environment to take effect. 

6. Activate the tensorflow environemnt:
    
        $conda acrtivate tensorflow_env

7. Install Keras (only one time)
        
        $conda install keras
        $pip install keras
    
    (you can do any one of them - I prefer the first one)
8. You are ready to go and try out the first scripts. 
