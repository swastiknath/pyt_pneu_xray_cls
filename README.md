# Constituting an End-to-End Deep Neural Network to precisely distinguish distinct types of Pneumonia using Chest X-Ray Images.

The dataset Version 2.0 and Version 3.0 for this model is made accessible by Daniel Karmany, Kang Zhang, Micheal Goldbaum et.al1 from Mendeley Data by Elsevier2, the dataset is associated to the University of California, San Diego and Guangzhou Women and Children’s Medical Centre. The data contains thousands of expert validated Chest X-Ray images and are split into Train and Test set of independent images, and each of these sets there is two more distributions named Normal and Pneumonia. In the distribution Normal, there are thousands of Pneumonia Negative X-ray images which is named with the patient’s id and type of the image class such as normal. In the Pneumonia distribution, we get other two types of images such as bacterial or viral contamination with their names as their representing labels appended with patient’s id. The dataset is completely anonymized, so there is no possible way to trace back to the patient from the images. To feed these data to a machine learning model we need to redistribute the images into properly configured directories. 

The whole dataset both Version 2.0 and Version 3.0 is available in the Mendeley Data website as zip files where version 2.0 is around 1.15 GB in size while the Version 3.0 is around 7.9 GB in size. 

So, in our AWS SageMaker Notebook instance, we download the file by copying the https link and issue a wget command which goes ahead and download the file into the AWS SageMaker Notebook Instance workspace. Next, we create a new directory in the notebook instance to hold the properly distributed directories. So, at first, we create two more directories inside the new directory to hold the train and test set. Now, at this stage, we use python libraries and control loops and distribute the images from Normal and Pneumonia directory into 3 separate directories which will work as the label of these images for both the training and test set. So, basically what we did was we changed the distribution of the images and put the images in their specific label directory like we put Bacteria infected Pneumonia X-Ray images into their directory. As we have chosen PyTorch as our preferred deep learning framework, the framework requires that the images are distributed into the same number of directories as there are target classes and treats the name of the directory as the label of the image. Next up, we shuffle up the dataset images and divide them into batches to feed our PyTorch based Machine Learning Model. In the process, we also resize them to properly fit the model’s requirements and convert them into multidimensional tensors. We use torchvision package to do so. PyTorch models perform the calculation on multidimensional tensors. We do not get into Image Augmentation process here as the dataset we have doesn’t vary much and the complex architectures of Deep CNNs, augmentation result into overfitting to the training data. We repeat the process for the test dataset too, and also devoid of any kind of image augmentation process. As we intend to deploy or model via the AWS SageMaker platform, to use the training and the test dataset with a SageMaker PyTorch Estimator, we need to upload the dataset already processed and properly distributed to an Amazon Simple Storage Service (S3) object in the same datacentre region as the model would be deployed to. 

### Downloading the Dataset:
We download the data from a reputable source in Scientific Research Community called [Mendeley by Elsevier ]('https://mendeley.com') the dataset was collected from real patients by different medical organizations around the world and was labelled by domain experts with utmost care and in the dataset archive website we see that they have said no corrupted files are there in the dataset at all. 
<hr> </hr>
<p>
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, v3
<a href="http://dx.doi.org/10.17632/rscbjbr9sj.3">http://dx.doi.org/10.17632/rscbjbr9sj.3</a> </p>
<p> 
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2
    <a href="http://dx.doi.org/10.17632/rscbjbr9sj.2">http://dx.doi.org/10.17632/rscbjbr9sj.2</a> </p>

<hr> </hr>

Obtaining the Dataset Version 2.0:
```
wget https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433/ChestXRay2017.zip?dl=1
from zipfile import ZipFile
with ZipFile('ChestXRay2017.zip', 'r') as dataobj:
    dataobj.extractall(path='####')
```
Obtaining the Dataset Version 3.0:

```
wget https://data.mendeley.com/datasets/rscbjbr9sj/3/files/810b2ce2-11c3-4424-996e-3bef36600907/ZhangLabData.zip?dl=1
from zipfile import ZipFile
with ZipFile('ZhangLabData.zip', 'r') as dataobj:
    dataobj.extractall(path='data')
```
