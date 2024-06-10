# Application of deep learning to classify thyroid cytology microscopic images according to The Bethesda system (TBS)
**Subject:**  Scientific Research Methods  
**Student(s):** Pham Ngoc Hai, QHT.2021 Data Science
**Supervisor(s)**: ASS Prof., Dr. Le Hong Phuong, Faculty of Mathematics - Mechanics - Informatics, HUS

## Abstract
**Objective:** Classify thyroid cell micrographs according to the Bethesda standard to develop a computer-aided diagnosis (CAD) system for doctors.  
**Method:** Use deep learning models to identify regions of interest in the images and classify the corresponding labels of the images.  
**Results:** The model was trained on a dataset of 7698 images and validated on a set of 954 images. Experimental results on 1491 images of the independent test set achieved an accuracy of up to 87%.  
**Conclusion:** The current research results demonstrate the potential of applying deep learning models in classifying thyroid cell micrographs according to the Bethesda standard. This lays the foundation for developing an efficient computer-aided diagnosis (CAD) system.

**Keywords:** Fine-Needle Aspiration Biopsy (FNAB) images, Artificial Intelligence (AI), Deep Learning (DL), Convolutional Neural Network (CNN)


## Video Illustration
<!-- [![Watch the video](https://img.youtube.com/vi/oCqzEb31S3o/maxresdefault.jpg)](https://youtu.be/oCqzEb31S3o)-->
<!-- when video in this repository  -->
<!-- [![Video Title](https://img.youtube.com/vi/oCqzEb31S3o/0.jpg)](https://www.youtube.com/watch?v=oCqzEb31S3o)  -->
<!-- when video from the YouTube-->
<div align="center">
  <a href="https://www.youtube.com/watch?v=oCqzEb31S3o"><img src="https://img.youtube.com/vi/oCqzEb31S3o/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>

Click the image above to watch the video on YouTube.

## Details about the report and slides:
See Report/ThyroidCancerReport.pdf & Report/ThyroidCancerSlide.pdf

## Important Highlights:

### Research Objective
The main objective of this study is to develop a computer-aided diagnosis (CAD) system for doctors in the clinical examination of thyroid biopsy samples taken by fine-needle aspiration.

To achieve this objective, the following tasks and problems need to be addressed:
- Automatically identify important regions in the images to reduce the observation time for doctors and to help the classification algorithm in the second objective produce faster predictions.
- Label biopsy images according to the corresponding levels based on the Bethesda standard (6 labels - however, due to the nature of the dataset, the current objective of the study is to classify 3 labels).
- Combine the first two objectives to build a supportive machine system.

## Dataset Description
This dataset consists of cell biopsy images taken from biopsy samples using the fine-needle aspiration method. This method is commonly used in cancer diagnosis due to its minimally invasive nature and ease of implementation.

### Data Source
The images were collected at the 108 Central Military Hospital when performing cell aspiration to diagnose thyroid cancer.

### Number and Characteristics of Images
The total number of images in the dataset is 1421, including 3 labels temporarily labeled as B2, B5, B6. These labels correspond to benign, suspicious for malignancy (SUS), and malignant in the Bethesda standard [[Evranos, I., et al. (2017)](https://doi.org/10.1111/cyt.12384)]. The number of images corresponding to labels B2, B5, B6 are 103, 541, and 777, respectively. The file format is JPEG with a resolution of 1224x960 pixels for label B6 and 1024x768 pixels for labels B2 and B5.

### Access and Usage
The dataset is currently restricted due to security reasons. Access and usage methods may be changed and updated along with new results of this research in the future.

### Data Collection Method
The images were collected using the fine-needle aspiration (FNAB) method, specifically:
- First, after an ultrasound, the patient is guided by the doctor to perform a cell biopsy to consider the possibility of surgery.
- The doctor continuously aspirates the suspected cancerous area with a fine needle under ultrasound guidance.
- The cells are then placed on a glass slide, with blood and non-essential cellular material removed.
- The cells on the slide are then stained with H&E (Hematoxylin and Eosin). Specifically:
  - Hematoxylin: An alkaline dye, stains the cell nucleus blue-purple.
  - Eosin: An acidic dye, stains the extracellular matrix and cytoplasm pink. Other structures may stain various shades and combinations of blue-purple and pink.
- After staining, the slide is placed under a microscope and photographed.

### Research Methods
There are 2 problems:
- The problem of determining the area of ​​interest in an image uses a deep learning model to identify individual cells combined with a clustering algorithm to cut out image areas with high cell concentration.
- The problem of classifying biopsy images FNAB uses a deep learning model to automatically extract important features of the image for classification.

### Results & Discussion
#### About the results achieved
- The cell identification model works well with a high mAP value of 72.8% on the test set, with good efficiency. With the clustering algorithm, the overall calculation of the entire set of 1421 images is 0.64, this is an acceptable threshold and the cropped image results are manually evaluated as good.
- With the image classification problem, it can be proven that cutting out important areas of the image instead of using the original image brings 10% higher results when tested with the H1 model, reaching 84% compared to 74%. %. By reducing the complexity of model H1, we obtain model H2. It can be seen that the H2 model gives 3% higher results on the test set, reaching 87% compared to 84%. This may be due to the simplicity of the model, which helps update the weights of the data more effectively.

#### About current goals and future directions
It can be said that the 3 goals of the research have been achieved, goal 1 supports goal 2 and goals 1 and 2 support goal 3. Selecting important areas in the image combined with classifying Corresponding labels for images are strongly supported by machines (CAD) that help doctors.
However, even after achieving the three set goals, the current study is still limited in that it has not evaluated the effectiveness of using this research to improve the diagnostic results of doctors - especially are inexperienced and experienced doctors.
Thus, the future goals of the research will be:
- Continue to improve the technique of cropping important image areas
- Improved the performance of the classification model 
- Especially research on the positive impact of this research result on the diagnostic effectiveness of doctors
