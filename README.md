<h1>Echo-Mamba: Mamba Network for Left Ventricle Segmentation in Echocardiography<h1>
--------------------------
<h4>Jieqiong Ding</br>
Northeast Forestry University,Heilongjiang, China<h4>



  
Dataset
-------

*A. Echonet-Dynamic* </br>
 1. Download the Echonet-Dynamic dataset from [this](https://echonet.github.io/dynamic/index.html) link. </br>
 2. The file format reference is as follows.
    - Echo-Mamba
      - a4c-video-dir
        - Videos      
        - VolumeTracings.csv
        - FileList.csv

*B. Echonet-Pediatric* </br>
 1. Download the Echonet-Pediatric dataset from [this](https://echonet.github.io/pediatric/index.html) link. </br>
 2. The file format reference is as follows.
    - Echo-Mamba
      - pediatric
        - A4C
          - Videos    
          - FileList.csv      
          - VolumeTracings.csv
        - PSAX
          - Videos 
          - FileList.csv   
          - VolumeTracings.csv
          
*C. CAMUS* </br>
 1. Download the CAMUS dataset from [this](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) link. </br>
 2. The file format reference is as follows.
    - Echo-Mamba
      - camus
        - image
        - mask      
        - train.txt
        - val.txt
        - test.txt
        
*D. HMC-QU* </br>
 1. Download the HMC-QU dataset from [this](https://www.kaggle.com/datasets/aysendegerli/hmcqu-dataset/data) link. </br>
 2. The file format reference is as follows.
     - Echo-Mamba
       - hmcqu
         - Videos 
         - mask      
         - train.csv
         - test.csv


Installation
------------
```
pip install . --use-feature=in-tree-build
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```


Train the EchoMamba
-----

    echonet segmentation --data_dir /your dir/ 


For more details, see the accompanying paper,
-----
> [Echo-Mamba: Mamba Network for Left Ventricle Segmentation in Echocardiography]<br/>

## Citation
If you find this repository helpful, please consider citing: </br>
```


```

