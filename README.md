# iAMPCN: a deep-learning approach for identifying antimicrobial peptides and their functional activities
## Introduction
Due to the microbial pathogens’ increasing resistance to chemical antibiotics, it is urgent to develop novel infectious therapeutics. 
Over the past decade, there have been several developments in utilizing antimicrobial peptides (AMPs) as potential alternatives to treat 
infections since most natural AMPs are particular polypeptide substances in living organisms and are critical components of the innate 
immune system which protects the host against invading pathogens. AMPs are generally small-molecule polypeptides and have diverse functional 
activities against target organisms such as bacteria, yeasts, fungi, viruses, and cancer cells. Compared with traditional chemical 
antibiotics, AMPs have higher antibacterial activities, broader antibacterial spectrums, and fewer possibilities resulting in target strains’ 
resistance mutation. Therefore, AMPs have a wide range of application prospects in the pharmaceutical industry and have become a hotspot 
of biomedical research.
Herein, we reviewed these computational approaches comprehensively, including the involved functional activities, benchmark datasets, machine 
learning algorithms, feature selection algorithms, and performance evaluation strategies and metrics. Then we developed a predictive 
framework named iAMPCN (identification of AMPs and their functional activities based on convolutional neural networks) and evaluated its 
ability to identify different kinds of functional activities of AMPs. The performance evaluation results demonstrated that iAMPCN achieved 
superior performances in identifying AMPs and their functional types compared with available predictive tools. In addition, we constructed 
an user-friendly web server based on this framework (http://iampcn.erc.monash.edu/) for the public to use. We sincerely hope iAMPCN serves 
as a prominent tool for identiying potential AMPs and their speicific functions that can be experimentally validated.
![image](https://user-images.githubusercontent.com/93033749/196317233-da4d5114-b32e-4df3-8f7c-08282a109cf5.png)
## Environment
* Ubuntu
* Anaconda
* python 3.8
## Dependency
* biopython                     1.79
* Flask                         2.1.2
* Flask-PyMongo                 2.3.0
* pandas                        1.4.2
* scikit-learn                  1.1.1
* scipy                         1.8.1
* torch                         1.12.1
* wheel                         0.37.1
* numpy                         1.23.1
* tqdm                          4.64.0
## Installation Guide
```
git clone https://github.com/joy50706/iAMPCN.git
conda create -n iampcn python==3.8
pip install numpy
pip install pandas
pip install biopython
pip install tqdm
pip install -U scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch
cd iAMPCN
```
## Usage
```
python predict.py -test_fasta_file {fasta file for predicting} -output_file_name {file name of prediction results}
```
For example:
* using the example test fasta file (examples/samples.fasta)
```
python predict.py -test_fasta_file examples/samples.fasta -output_file_name prediction_results
```
or 
```
python predict.py -test_fasta_file {fasta file for predicting} -output_file_name {file name of prediction results}
```
* using other files, just change the file name, the final prediction results will be saved using the '.csv' format.
the predictive output file looks like this: 
![image](https://user-images.githubusercontent.com/93033749/196341873-2d8adec5-7464-4eb8-bd1d-345ec00902d8.png)
## Reference
