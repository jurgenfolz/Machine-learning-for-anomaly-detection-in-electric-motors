# Machine learning for anomaly detection in electric motors
 A study of machine learning algorithms for anomaly detection in electric motors by vibration and audio signals

Authors: Klaus Jürgen Folz, Herbert Martins Gomes<br />
Universidade Federal do Rio Grande do Sul, Departament of Mechanical Engineering<br />
Data available in https://www02.smt.ufrj.br/~offshore/mfs/page_01.html

ABSTRACT <br />
Machine maintenance requires continuous improvement of techniques and 
equipment for monitoring equipment operating parameters. The financial gains in 
avoiding catastrophic and cascading problems in industrial plants due to failures far 
outweigh the expenses with monitoring investments in new technologies. The use of 
machine learning techniques has become an area of intense research, largely due to the 
relative success of these methodologies in defect classification, lifetime predictions, and 
the possibility of an online and real-time monitoring. The objective of this article is to 
evaluate and compare the performance of two machine learning algorithms, Support 
Vector Machine (SVM) and Random Forests (RF), when classifying 7 states of operation 
of an electric motor using the Mel Frequency Cepstral Coefficients (MFCCs) as features, 
calculated using the motor’s vibration and audio signals separately. After the training, the 
SVM model obtained a mean accuracy of 100 % for the MFCCs obtained from the 
vibration signals and 69.6% for the audio signal. The RF had a mean accuracy of 99.15% 
for the MFCCs from the vibration signals and 63.82% for the audio signal.

1 INTRODUCTION <br />
Modern electric motors are found in the most diverse activities, in which force and 
torque are required to move turbines, boilers, conveyors and other components. Problems 
such as misalignment, imbalance and coupling issues can lead to dangerous conditions or 
even to the equipment’s premature failure. The precise identification of malfunctions 
makes a premature correction possible and, therefore, an extension of the machine’s 
lifetime, or even avoid more catastrophic events or frequent stops for repair. 
The efficiency and reliability of electric motors are of paramount importance in many 
industrial applications. The failure of such equipment can be a serious problem in terms 
of safety or costs. To achieve adequate results, predictive maintenance methods can be 
used, which, through constant condition monitoring, enable the prediction and detection 
of failures. Predictive maintenance practices, when compared to traditional maintenance 
methods, increase the average time between each revision, reduce the stock of spare parts 
and minimize downtime for corrective maintenance [3]. Condition monitoring techniques 
generate large amounts of data, requiring the automation of diagnostic methods. If 
properly trained, machine learning algorithms provide an alternative for autonomously 
interpreting data. 
The objective of this article consists of comparing the performance of the Support 
Vector Machine (SVM) and Random Forest (RF) algorithms, when classifying 7 states 
of operation of a ¼ hp electric motor (1 normal and 6 anomalous). For that reason, the 
Mel Frequency Cepstral Coefficients (MFCCs) will be used as features. The coefficients 
are calculated from the vibration signals of two 3-channel accelerometers installed in a 
shaft bearing and using the operating motor’s audio signal. The states are vertical 
misalignment, horizontal misalignment, cage fault, ball fault, outer race, imbalance, and 
the normal state of operation. 
Furthermore, the results obtained for both algorithms with the coefficients calculated 
from the audio and vibration signals separately will be compared, to determine if it is 
possible to use only the audio signal, since it’s a less costly and complex way to acquire 
the data. The performance evaluation will be based on the Receiver Operating 
Characteristic Curve (ROC Curve), Mean Accuracy, F1 Score and Confusion Matrices. 
The data was made available by the Laboratory of Signals, Multimedia and 
Telecommunications (SMT) of the Federal University of Rio de Janeiro (UFRJ) in the 
Machinery Fault Database [9]. 
Machine Learning (ML) is the generic term commonly used to designate a series of 
techniques and algorithms that can classify and acquire information from data. [20] 
evaluated the performance of the Support Vector Machines and Fast Clustering 
Algorithm, combined with the variational mode decomposition method and principal 
component analysis, in predictive maintenance of imbalanced rotating machines. The 
authors concluded that the model would be able to effectively diagnose failures caused 
by imbalance in rotating machines. 
A prediction model for proactively preventing failures in a gravitational accelerator 
was proposed by [7], using a 4-channel accelerometer coupled to the bearing housing. 
The algorithm was trained with a Deep Learning model, converting the vibration signal 
into a short-term Fourier transform (STFT) and into a spectrogram of Mel Frequency 
Cepstral Coefficients (MFCCs) converting the result into a two-dimensional image. As a 
result, it was possible to quantify the severity of the imbalance by observing the defect 
areas that cannot be seen with one-dimensional images. 
[13] investigated different statistical methods of machine learning and control graphics 
for the automatic detection of anomalies in a rotating bearing from a commercial 
semiconductor manufacturing machine. The developers showed that both the control 
charts and the fully supervised classification methods performed very similarly, directly 
conditioned to the quality of the training data. 
An unsupervised machine learning model for condition monitoring of rotating 
machines using an anomaly detection approach was studied by [1]. Effectiveness was 
evaluated by comparing the automatically generated results with a manual dataset by 
training an Isolation Forest model achieving an average F1-score of 99.6%.<br />

2 THEORETICAL BACKGROUND <br />
2.1 Support Vector Machine (SVM) <br />
SVM is a supervised learning classifier, used both in classification and for regression 
analysis, clustering and other machine learning applications using a geometric concept 
called hyperplane (decision limit). The purpose of the SVM algorithm is to compute a 
hyperplane in an N-dimensional space (where N is the number of features) to distinctly 
classify (separate) the data points. However, for a given dataset, there are many different 
hyperplanes that can successfully separate the classes. To determine the best possible 
plane, the algorithm will maximize the separation of the hyperplane and its closest points 
(features), also called margin. The robustness of the algorithm is directly linked to the 
need for the separation between the points and the hyperplane being as large as possible 
for all classes. Therefore, the objective becomes to calculate the plane that has the 
maximum distance between the data points of the classes.

2.2 Random Forests <br />
[5] devised and developed the first Random Forests (RF) algorithm using a random 
subspace method. Later the model was improved by Leo Breiman and Adele Cutler [4]. 
The random forest algorithm is defined by [4] as a combination of decision tree predictors, 
where each tree depends solely on the values of an independent input dataset with the 
same distribution for all trees in the set (forest). In this way, a random subset of a certain 
size is produced from the space of possible attributes (input data) of division. Therefore, 
the best split is the deterministically selected feature of this subset. For the 
classification task, the algorithm classifies the instance by combining all the results from 
each of the trees in the forest. In other words, the method used to combine the results can 
be as simple as predicting the class obtained from the largest number of trees, as in a 
majority vote. [21] highlights the ease of parameterization of the model, so that different 
values for the parameters have little influence on performance and accuracy. However, 
different values for the total number of trees used in the forest and the maximum depth 
of each tree are commonly tested for in-depth assessments of the model's accuracy. 


2.3 Mel Frequency Cepstral Coefficients (MFCCs) <br />
Mel Frequency Cepstral Coefficients (MFCCs) are attributes extracted from a wave 
signal, widely applied in speech recognition algorithms. [8] interprets the coefficients as 
a compact description of the shape of a spectral envelope of a signal. The MFCC feature 
extraction technique, according to [6] consists of splitting the signal into windows, 
calculating the Discrete Fourier Transform (DFT), obtaining the logarithm of magnitude 
and the distortion of frequencies in a Mel scale, followed by the Inverse Discrete Cosine 
Transform (DCT). The detailed description of the various steps involved in extracting the 
MFCC is explained below.<br />

    1. Splitting the signal into windows: The analysis should always be performed in short 
    segments in which the signal is approximately stationary. 
    2. Frequency Spectrum (DFT): Each segment is converted into a magnitude spectrum 
    by applying DFT. 
    3. Mel’s Spectrum: The Spectrum is calculated by passing the Fourier transformed 
    signal through a set of bandpass filters known as the Mel filter bank. 
    4. Filter Bank: The filter center frequencies are normally evenly spaced on the 
    frequency axis
    5. Discrete cosine transform: The DCT is applied to the transformed Mel frequency 
    coefficients and produces a set of cepstral coefficients, the MFCCs.


3 DATA PROCESING<br />
A single measurement represents 5 seconds for each class, representing 250,000 
points in time. Thus, the 1951 files with their 487.75 million rows and 8 columns were 
iteratively imported to calculate the MFCCs. To save RAM memory, the vibration signal 
data in the time domain is then discarded, only the MFCCs are stored in a dictionary type 
variable. The feature.mfcc function of the Librosa package for Python was used to obtain 
the MFCCs, with a number of coefficients equal to 40, resolution of the fast Fourier 
transform equal to 2048, and each data set was segmented into 5 equal parts, resulting in 
a subset of 1 second per frame superimposed by 512 measurements or 0.01024 seconds. 
The resulting set of MFCCs is a two-dimensional 40-row by 98-column array 
(numpy array in Python) for each 5-second dataset from a single measurement. The 
application of the SVM and Random Forests algorithms requires a one-dimensional set 
of features. In order to circumvent this limitation, the resulting matrix of the MFCCs 
undergoes a juxtaposition of all its lines, transforming the original 40 x 98 matrix into a 
one-dimensional vector with 3920 elements for each measurement using the 
numpy.flatten method of the Numpy package for Python. Finally, for a single vibration 
measurement, the result is a matrix of 1951 rows and 3921 columns, one of which 
contains the description of the fault or faultless normal state and the other 3920 the 
respective MFCCs juxtaposed to a single axis (axial, tangential or radial) or for the audio 
signal. 
To use all measured axes and encompass both accelerometers (upper and lower) 
and not lose any information from the original signals, the procedure of juxtaposing the 
lines is repeated. However, in this step the juxtaposition is applied on the cepstral 
coefficients calculated for each axis (axial, tangential and radial) for each accelerometer. 
The result is a one-dimensional vector with 23,520 features, representing all MFCCs 
calculated for the signals of all axes. The procedure is not repeated for the audio signal, 
as this is just one measurement. 
Finally, the data are randomly divided into two subsets: test and training. The 
random state is fixed, allowing reproducibility of tests and validations and comparison of 
results for different algorithms and different iterations. The training data represents 70% of the original set, that is, 1,366 samples 
containing 32,128,320 coefficients calculated from the vibration signals and 5,354,720 
for the audio signal. The test data contains 585 different samples, 30% of the original set, 
with 13,759,200 MFCCs for vibration signals and 2,293,200 for the sound wave. <br />

4 DISCUSSION <br />
The SVM model with the polynomial Kernel function, degree (d) and constant (r) 
equal to 3 and regularization parameter (C) equal to 10 provided an error-free 
classification with 100% average accuracy for the MFCCs calculated from the vibration 
signals. The confusion matrix, shown in Figure 2 - SVM Confusion matrix with MFCCs 
calculated using vibration (a) and audio signals (b). (a), does not have a single off-
diagonal element, which is characteristic of a perfect classifier. Likewise, the ROC curve 
with parameter AUC equal to 1, shown in Figure 3 (a), is identical to the curve of a perfect 
classifier. 
For the set of MFCCs obtained from the audio signal, the F1 scores for all classes of 
anomalies were close to a random classifier, except for the vertical misalignment class, 
with 91.4% accuracy. The lowest F1 score, belonging to the normal class (24%), is due 
to the low number of samples for this class, with only 49 measurements. The model 
training lasted 33 seconds, approximately 37% faster when compared to the training 
performed by the SVM with MFCCs from the vibration signals. The higher speed in the 
training stage is influenced by the lower number of attributes for the model built with the 
audio signal, with 3920 coefficients against 23520 coefficients for the model built from 
the vibration signal. 
The RF’s 99.15% mean accuracy shows that, even with the best performance of the 
SVM algorithm, the RF algorithm still is an excellent classifier. Likewise, the model's 
ROC curve approaches the curve of a perfect classifier, being compromised only by the 
errors made in the normal class, with AUC of 0.9996. 
For both algorithms, the performance for the set of MFCCs generated from the 
vibration signals and the audio signal differ substantially. While the vibration data and 
their respective cepstral coefficients provide the creation of machine learning models very 
close to perfect classifiers, the model trained with the coefficients calculated using the 
audio signals have a behavior like random classifiers. Furthermore, all metrics presented 
in this article point to the same result, with the SVM algorithm being the best classifier 
in terms of classification performance and processing time in the model training stage. 
7 CONCLUSION 
The Support Vector Machine (SVM) using the polynomial Kernel function, with 
degree (d) and constant (r) equal to 3 is the best option for classifying problems and 
anomalies in electric motors using the Mel frequencies cepstral coefficients as features. 
The Algorithm showed average accuracy and F1 score of 100% for all classes, for MFCCs 
obtained via vibration signals and average accuracy of 69.6% for coefficients calculated 
from the sound signal. In addition, the SVM model proved to be faster in the training 
phase when compared to the Random Forests model, with a duration of 51 seconds against 
1 minute and 17 seconds of the concurrent algorithm. This result is due to the high number 
of attributes, MFCCs, and the correct choice of parameters to obtain the best results. 
The MFCCs were excellent attributes for applications of fault detection in electric 
motors using vibration signals. However, applying the same methods and parameters, 
with the coefficients calculated from the motor’s audio signal (very similar application 
for which MFCCs were initially developed), both models did not perform well, with a 
better mean accuracy of 69.6% from the SVM model. Although the results obtained via 
audio signal were not adequate, the application of models of this type in real situations in 
the industry would be less complex and of lower cost, when compared to the application 
using accelerometers, due to the cost of this equipment and the complexity of their 
configuration in a machine.<br />

5 REFERENCES <br />
[1] AHMADs S.; STYP-REKOWSKI, K.; NEDELKOSKI, S; O. KAO, Autoencoderbased Condition Monitoring and Anomaly Detection Method for Rotating 
Machines. 2020 IEEE International Conference on Big Data (Big Data), 2020, p. 
4093-4102. https://doi.org/10.48550/arXiv.2101.11539 <br />
[2] ANTONI, L.; ZIEMOWIT, D.; PIOTR, C. An anomaly detection method for 
rotating machinery monitoring based on the most representative data. Journal of 
Vibroengineering Vol. 23, Issue 4, 2021, p. 861-876. 
https://doi.org/10.21595/jve.2021.21622 <br />
[3] ARALTO, A. Manutenção Preditiva: Usando Análise de Vibrações. São Paulo: 
Manole, 2004. <br />
[4] BREIMAN, L. Random Forests. Machine Learning, v. 45, p. 5–32, 2001. 
https://doi.org/10.1023/A:1010933404324 <br />
[5] HO, TIN KAM (1995). Random Decision Forests. Proceedings of the 3rd 
International Conference on Document Analysis and Recognition. Montreal, QC, 14–
16 August 1995. pp. 278–282. https://doi.org/10.1109/ICDAR.1995.598994 <br />
[6] K.S. RAO AND MANJUNATH K.E., Speech Recognition Using Articulatory and 
Excitation Source Features. Springer Briefs in Speech Technology, 2017. 
https://doi.org/10.1007/978-3-319-49220-9 <br />
[7] LEE, S.; YU, H.; YANG, H.; SONG, I.; CHOI, J.; YANG, J.; LIM, G.; KIM, K.-S.; 
CHOI, B. Kwon, J. A Study on Deep Learning Application of Vibration Data and 
Visualization of Defects for Predictive Maintenance of Gravity Acceleration 
Equipment. Appl. Sci. vol. 11, p. 1564, 2021. https://doi.org/10.3390/app11041564 <br />
[8] LERCH, A. An introduction to audio content analysis: Applications in signal 
processing and music informatics. [s.l.] Wiley-IEEE Press, 2012. 
https://doi.org/10.1002/9781118393550 <br />
[9] MAFAULDA. COPPE/Poli/UFRJ. (2014). Machinery Fault Database – 
MAFAULDA. Available: http://www02.smt.ufrj.br/~offshore/mfs/page_01.html. 
Access in: 21/09/2021. <br />
[10] MCFEE, B. et al. librosa/librosa: 0.8.0. 22 July 2020. 
https://doi.org/10.25080/Majora-7b98e3ed-003 <br />
[11] PEDERSEN, R.; SCHOEBERL, M. (2006). An Embedded Support Vector 
Machine. In Proceedings of the Fourth Workshop on Intelligent Solutions in 
Embedded Systems. WISES, 2006, pp. 79-89. 
http://hdl.handle.net/20.500.12708/51591 <br />
[12] PEDREGOSA ET AL. Scikit-learn: Machine Learning in Python. Journal of 
Machine Learning Research, v. 12, p. 2825–2830, 2011. http://scikitlearn.sourceforge.net <br />
[13] PITTINO, F., PUGGL M., MOLDASCHL, T., HIRSCHL, C. Automatic Anomaly <br />
Detection on In-Production Manufacturing Machines Using Statistical Learning 
Methods. Sensors. v. 20, p. 8: 2344, 2020. https://doi.org/10.3390/s20082344 <br />
[14] PRATI, R. C.; BATISTA, G. E. A. P. A; MONARD, M. C., Evaluating Classifiers 
Using ROC Curves. IEEE Latin America Transactions, vol. 6, no. 2, pp. 215-222, 
June 2008. https://doi.org/10.1109/TLA.2008.4609920 <br />
[15] PYTHON. Support Vector Machine Python Example. Towards data science. 
Access in: 15/07/2021. Available: https://towardsdatascience.com/ support-vectormachine-python-example-d67d9b63f1c8 <br />
[16] SCIKIT. Support Vector Machines – Scikit-Learn Documentation. Available: 
https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation. Access 
in: 11/10/2021. <br />
[17] SIRIKULVIRIYA, NAPHAPORN AND SUKREE SINTHUPINYO. Integration of 
Rules from a Random Forest. International Conference on Information and 
Electronics Engineering IPCSIT vol.6, 2011. <br />
[18] SpectraQuest. SpectraQuest, Inc. Available: https://spectraquest.com/ Acess in: 
21/09/2021. <br />
[19] VAPNIK, V. N. The Nature of Statistical Learning Theory. Springer, NY, 1995. 
https://doi.org/10.1007/978-1-4757-2440-0 <br />
[20] ZHANG, X.; JIANG, D.; HAN, T.; WANG, N.; YANG, W.; YANG, Y. Rotating 
Machinery Fault Diagnosis for Imbalanced Data Based on Fast Clustering 
Algorithm and Support Vector Machine. Journal of Sensors, Beijing, China, v. 
2017, p.0-15, 2017. https://doi.org/10.1155/2017/8092691 <br />
[21] HORNING, N. Random Forests: An algorithm for image classification and 
generation of continuous fields data sets, 2010. <br />
[22] GAVRISHCHAKA, V.; GANGULI, S. Support vector machine as an efficient tool 
for high-dimensional data processing: Application to substorm forecasting. 
Journal of Geophysical Research. 106. 29911-29914, 2001. 
https://doi.org/10.1029/2001JA900118 <br />
[23] SCIKIT. Metrics and scoring: quantifying the quality of predictions. Available: 
https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics. Access 
in: 07/11/2021.<br />