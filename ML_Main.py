# A STUDY OF MACHINE LEARNING ALGORITHMS FOR ANOMALY DETECTION IN ELECTRIC MOTORS BY VIBRATION AND AUDIO SIGNALS
#Klaus Jürgen Folz, Herbert Martins Gomes
#Universidade Fedral do Rio Grande do Sul (UFRGS)
# Data avaiable in https://www02.smt.ufrj.br/~offshore/mfs/page_01.html

"""Machine maintenance requires continuous improvement of techniques and 
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
"""

#Packages
import pandas as pd
import os
import numpy as np
import datetime
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import math
import json
import matplotlib
import random
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import joblib
import pickle
import warnings
warnings.simplefilter("ignore")


class MODEL():
    
    def __init__(self,SAMPLE_TIME,SAMPLE_RATE,DIR_MAIN,
                 N_MFCC,N_FFT,HOP_LENGTH,NUM_SEGMENTS,
                 RANDOM_STATE_,kernel_grid_search,
                 c_grid_search,coef0_grid_search,degree_grid_search):
        
        self.SAMPLE_TIME=SAMPLE_TIME 
        self.SAMPLE_RATE=SAMPLE_RATE 
        self.DIR_MAIN=DIR_MAIN
        self.N_MFCC=N_MFCC
        self.N_FFT=N_FFT
        self.HOP_LENGTH=HOP_LENGTH
        self.NUM_SEGMENTS=NUM_SEGMENTS
        self.SAMPLES_PER_SEGMENT=int(SAMPLE_RATE*SAMPLE_TIME/NUM_SEGMENTS)
        self.NUM_MFCC_PER_SEG=math.ceil(self.SAMPLES_PER_SEGMENT/HOP_LENGTH)
        self.RANDOM_STATE_=RANDOM_STATE_
        self.dict_with_data={
                            'label':[],
                            'mfcc' : [],
                            'index': [],
                            'mfcc_mic' : []
                        }
        
        #SVM parameters:
        self.kernel_grid_search=kernel_grid_search
        self.c_grid_search=c_grid_search
        self.coef0_grid_search=coef0_grid_search
        self.degree_grid_search=degree_grid_search
         
    @staticmethod
    def diftime(start,end):
        """diftempo - Calculates the time between two inputs, start and end
     Args:
         start(datetime): Start
         end(datetime): end
    """
        dif = end-start
        seconds_in_day = 24 * 60 * 60
        datetime.timedelta(0, 8, 562000)
        dif=divmod(dif.days * seconds_in_day + dif.seconds, 60)
        return dif
    
    @staticmethod
    def avm_f1(am,f1_score,labels,model):
        """avm_f1 - Prints ML model results: Average Accuracy and F1 Score
            Args:
                am (float): Average accuracy
                f1_score (list): List
                labels (list):labels
                modelo (str): the model title
        """
        
        print(f'Average accuracy {model}: {"{0:.2%}".format(am)}')
        result=[]
        for x in f1_score:
            result.append("{0:.2%}".format(x))
        #Exibe na Tela Resultados
        print(f'\n{model} - F1 score: ')
        i=0
        while i<len(labels):
            print(f'{labels[i]} : {result[i]}')
            i+=1
    
    @staticmethod   
    def confusion_matrix(title,y_pred,labels_test,labels):
        """confusion_matrix - Plots and saves the confusion matrix
            Args:
                title (str): title for the figure
                y_pred (list): predictions for the test data
                labels_teste (list): the lables for y_pred
                labels (list): the original labels
            """
        #Inputs da matriz
        fig, ax = plt.subplots(figsize=(10,6))
        fig.suptitle(title, fontsize=14,color='black')
        #Dados Matriz
        cm = confusion_matrix(labels_test, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        sn.heatmap(df_cm, annot=True, fmt='d')
        plt.tight_layout()
        plt.savefig(title+' MCONF'+'.png')
    
    
    #Functions
    def roc_curve(self,model,labels_test,model_name,source='vib',lw = 2):
        
        if source=='vib':
            mfccslist=self.mfccs_list
        elif source=='audio':
            mfccslist=self.mfccs_mic_list
            
        y = label_binarize(self.labels_list, classes=list(dict.fromkeys(labels_test)))
        n_classes = y.shape[1]

        #Train and Test split para as labels agora em formato binário
        X_train, X_test, y_train, y_test = train_test_split(mfccslist, y, test_size=.3,
                                                            random_state=RANDOM_STATE_)

        # Treino para que cada classe seja comparada com todas as outras
        classifier = OneVsRestClassifier(model)
        try:
            y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        except:
            y_score=classifier.fit(X_train, y_train).predict(X_test)

        # Calcula a curva ROC e a área baixo da curva ROC para cada classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(10,6))
        plt.plot(fpr["micro"], tpr["micro"],
                label='Micro average ROC curve (AUC = {0:0.4f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='Macro average ROC curve (AUC = {0:0.4f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darkolivegreen','coral','slategrey','darkmagenta'])
        for i, color in zip(range(n_classes), colors):
            lb_graf=list(dict.fromkeys(labels_test))[i]
            area = "{0:.4f}".format(roc_auc[i])
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label=f'ROC curve for {lb_graf} (AUC = {area})')

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve {model_name} ({source})')
        plt.legend(loc="lower right")
        plt.savefig(f'ROC curve {model_name} ({source})'+'.png')
    
    def __data_adjustment(self,method='avg'):
        """Function for data adjustment:
        - Method (string) - default ='media' - MFCCS formatting method:
            - 'average' -> makes the average of the mfccs transforming into a 1D array
            - 'juxtapose' -> makes the juxtaposition of the lines
        """
        mfccs_list=[] 
        labels_list=[] 
        mfccs_mic_list=[] 
        
        for mfcc,lb,mfcc_mic in zip(np.array(self.dict_with_data['mfcc']),self.dict_with_data['label'],np.array(self.dict_with_data['mfcc_mic'])):
            
            if lb.find('horizontal-misalignment')!=-1:
                lb='Horizontal misalignment'
            elif lb.find('imbalance')!=-1:
                lb='Imbalance'    
            elif lb.find('normal')!=-1:
                lb='Normal'
            elif lb.find('overhang-ball_fault')!=-1 or lb.find('underhang-ball_fault')!=-1:
                lb='Ball fault'
            elif lb.find('underhang-cage_fault')!=-1 or lb.find('overhang-cage_fault')!=-1:
                lb='Cage Fault'
            elif lb.find('vertical-misalignment')!=-1:
                lb='Vertical misalignment'
            elif lb.find('overhang-outer_race')!=-1 or lb.find('underhang-outer_race')!=-1:
                lb='Outer race'

            #Method to tranform the data
            if method=='avg':
                mfccs_list.append(np.mean(mfcc.T,axis=0)) #Avg of the mfccs
                mfccs_mic_list.append(np.mean(mfcc_mic.T,axis=0)) 
            elif method=='juxtaposition':
                mfccs_list.append(np.array(mfcc.flatten())) 
                mfccs_mic_list.append(np.array(mfcc_mic.flatten())) 
                
            np.array(mfcc.flatten()).shape
            labels_list.append(lb)

            
            
        return mfccs_list,labels_list,mfccs_mic_list 
    
    def get_mfccs(self):
        """Reading Data (8 columns) 
        Speedometer
        UBA - underhang bearing accelerometer (axial, radiale tangential direction)
        OBA - overhang bearing accelerometer (axial, radiale tangential direction)
        Microphone"""
        #Loop in the main folder
        for i,(dirpath, dirnames, filenames) in enumerate(os.walk(self.DIR_MAIN)):
            #Makes sure that the loop is in the correct folder
            if dirpath is not self.DIR_MAIN:
                #Saves the subfolder names, the Label
                dirpath_comp=dirpath.split('\\')
                pasta_atual=f'{dirpath_comp[-3]}-{dirpath_comp[-2]}-{dirpath_comp[-1]}'
                print(f'READING {pasta_atual.upper()}')
                for f in filenames:
                    #Load the file
                    file_pth=os.path.join(dirpath,f)
                    sr=self.SAMPLE_RATE
                    print(f'\tFile:{file_pth}')
                    signal_df=pd.read_csv(file_pth,delimiter = ",",header=None)
                    signal_df=signal_df.rename(columns={0: "vel_signal", 1: "UBA_axial", 
                            2: "UBA_radial",3: "UBA_tangential", 
                            4: "OBA_axial", 5: "OBA_radial",
                            6: "OBA_tangential", 7: "mic_signal"})
                    
                    
                    #-----------------------------------------------------------------------------------------------------------------------#
                    #-----------------------------------------------------------------------------------------------------------------------#
                    #----------------------------------------------------------MFCCS--------------------------------------------------------#
                    #-----------------------------------------------------------------------------------------------------------------------#
                    #-----------------------------------------------------------------------------------------------------------------------#
                    
                    #Loop in the columns
                    mfcc=[]
                    mfcc_mic=[]
                    for col in signal_df.columns:
                        col=str(col) #col title as string
                        if col!='vel_signal' and col!='mic_signal': #just the relevant columns
                            for s in range(self.NUM_SEGMENTS):
                                start=self.SAMPLES_PER_SEGMENT*s
                                end=start+self.SAMPLES_PER_SEGMENT 
                                signal=signal_df[col].to_numpy() #signal to numpy array
                                #Gravar os MFCCs para o segmento, se ele conter o tamnho correto de MFCC
                                mfcc1=librosa.feature.mfcc(signal[start:end],
                                                            sr=self.SAMPLE_RATE,
                                                            n_fft=self.N_FFT,
                                                            n_mfcc=self.N_MFCC,
                                                            hop_length=self.HOP_LENGTH)
                                
                            if mfcc1.shape[1]==self.NUM_MFCC_PER_SEG:
                                mfcc.append(mfcc1)
                        #for the MIC Signal
                        elif col=='mic_signal':
                            for s in range(self.NUM_SEGMENTS):
                                start=self.SAMPLES_PER_SEGMENT*s
                                end=start+self.SAMPLES_PER_SEGMENT 
                                sinal=signal_df[col].to_numpy()
                                mfcc_mic1=librosa.feature.mfcc(sinal[start:end],
                                                            sr=self.SAMPLE_RATE,
                                                            n_fft=self.N_FFT,
                                                            n_mfcc=self.N_MFCC,
                                                            hop_length=self.HOP_LENGTH)
                                
                            if mfcc_mic1.shape[1]==self.NUM_MFCC_PER_SEG:
                                mfcc_mic.append(mfcc_mic1)
                            
                                
                    self.dict_with_data['mfcc'].append(mfcc) #MFCCs for the vibration signals
                    self.dict_with_data['index'].append(i-1) #Index
                    self.dict_with_data['label'].append(pasta_atual) #The labels
                    self.dict_with_data['mfcc_mic'].append(mfcc_mic) #MFCCs for the mic signal
                    
                    self.mfccs_list,self.labels_list,self.mfccs_mic_list=self.__data_adjustment(method='juxtaposition')
                    self.labels = list(set(self.labels_list))
                    
                   
    def train_svm(self,source='vib'):
        
        if source=='vib':
            mfccs_train,mfccs_test,labels_train,labels_test=train_test_split(self.mfccs_list,
                                                                     self.labels_list,
                                                                     test_size=0.3,
                                                                    random_state=self.RANDOM_STATE_)
        elif source=='audio':
            mfccs_train,mfccs_test,labels_train,labels_test=train_test_split(self.mfccs_mic_list,
                                                                     self.labels_list,
                                                                     test_size=0.3,
                                                                    random_state=self.RANDOM_STATE_)
        
        #List for time and performance data
        cols_svm=[]
        #Loop for grid search
        for kernel in kernel_grid_search:
            for C in c_grid_search:     
                if kernel=='poly': #if the kernel function is polynomial, code loops using the coeficients and degrees
                    for coef0 in coef0_grid_search:
                        for degree in degree_grid_search:
                            start_svm=datetime.datetime.now()
                            #Support Vector Machines
                            clf = svm.SVC(C=C, kernel=kernel, gamma='auto',probability=True,coef0=coef0,degree=degree)
                            clf.fit(mfccs_train,labels_train) #SVM
                            #Average Accuracy
                            am_svm=(clf.score(mfccs_test, labels_test))
                            _f1_svm=f1_score(labels_test, clf.predict(mfccs_test), average=None, labels=self.labels) #f1
                            y_pred_svm = clf.predict(mfccs_test) 
                            end_svm=datetime.datetime.now() #dend time

                            dif_time_svm=self.diftime(start_svm,end_svm) #
                            time=f'{dif_time_svm[0]} min {dif_time_svm[1]} s'

                            cols_svm.append([C,kernel,am_svm,coef0,degree,dif_time_svm[0],dif_time_svm[1]])

                            print(f'SVM: C={C} Kernel={kernel} - Accuracy {am_svm} - Coef0 {coef0} - Degree {degree} - Time to train the model: {time}')
                            self.avm_f1(am_svm,_f1_svm,self.labels,'Support Vector Machines (SVM)')
                            self.confusion_matrix(f'SVM ({source}) C={C} Kernel={kernel}',y_pred_svm,labels_test,self.labels)
                            
                else: #Case it's a different kernel function
                    coef0=0
                    degree=3
                    start_svm=datetime.datetime.now()
                    #Support Vector Machines
                    clf = svm.SVC(C=C, kernel=kernel, gamma='auto',probability=True,coef0=coef0,degree=degree)
                    clf.fit(mfccs_train,labels_train) #SVM
                    #Average Accuracy
                    am_svm=(clf.score(mfccs_test, labels_test))
                    _f1_svm=f1_score(labels_test, clf.predict(mfccs_test), average=None, labels=self.labels) #f1
                    y_pred_svm = clf.predict(mfccs_test) 
                    end_svm=datetime.datetime.now() #end of the train

                    dif_time_svm=self.diftime(start_svm,end_svm) #time to train
                    time=f'{dif_time_svm[0]} min {dif_time_svm[1]} s'

                    cols_svm.append([C,kernel,am_svm,coef0,degree,dif_time_svm[0],dif_time_svm[1]])

                    print(f'SVM: C={C} Kernel={kernel} - Accuracy {am_svm} - Coef0 {coef0} - Degree {degree} - Time to train the model: {time}')
                    self.avm_f1(am_svm,_f1_svm,self.labels,'Support Vector Machines (SVM)')
                    self.confusion_matrix(f'SVM ({source}) C={C} Kernel={kernel}',y_pred_svm,labels_test,self.labels)
                    
        self.SVM=clf
        self.roc_curve(self.SVM,labels_test,'SVM',source)
        return self.SVM
        
    def train_random_forests(self,source='vib'):
        
        if source=='vib':
            mfccs_train,mfccs_test,labels_train,labels_test=train_test_split(self.mfccs_list,
                                                                     self.labels_list,
                                                                     test_size=0.3,
                                                                    random_state=self.RANDOM_STATE_)
        elif source=='audio':
            mfccs_train,mfccs_test,labels_train,labels_test=train_test_split(self.mfccs_mic_list,
                                                                     self.labels_list,
                                                                     test_size=0.3,
                                                                    random_state=self.RANDOM_STATE_)
        
        #Random Forests
        cols_rf=[]
        for depth in m_depth:
            for n_est in n_estimators:
                start_rf=datetime.datetime.now()
                clf_rf = RandomForestClassifier(n_estimators=n_est,max_depth=depth, random_state=0)
                clf_rf.fit(mfccs_train, labels_train)
                #Average acc
                am_rf=clf_rf.score(mfccs_test, labels_test)
                _f1_rf=f1_score(labels_test, clf_rf.predict(mfccs_test), average=None, labels=self.labels)
                y_pred_rf = clf_rf.predict(mfccs_test)
                end_rf=datetime.datetime.now() #end of the training
                dif_tempo_rf=self.diftime(start_rf,end_rf) #time to train
                time_rf=f'{dif_tempo_rf[0]} min {dif_tempo_rf[1]} s'
                cols_rf.append([n_est,depth,am_rf,dif_tempo_rf[0],dif_tempo_rf[1]])
                print(f'RF: Max Depth={depth} Trees={n_est} - Accuracy {am_rf} - Time to train the model: {time_rf}')   
                
                self.avm_f1(am_rf,_f1_rf,self.labels,'Support Vector Machines (SVM)')
                self.confusion_matrix(f'RF ({source}) Max Depth={depth} Trees={n_est}',y_pred_rf,labels_test,self.labels)
                
        self.RF=clf_rf
        self.roc_curve(self.RF,labels_test,'RF',source)
        return self.SVM


if __name__=='__main__':
    SAMPLE_TIME=5 #Sample time in seconds
    SAMPLE_RATE=50000 #rate of data acquisition, in this case the accelerometer or mic rates
    DIR_MAIN=r'C:\Users\nini_\Desktop\Hauptordner\Dados' #Wehere are the data
    N_MFCC=40 #Number of MFCCs
    N_FFT=2048 #NBits for the fast fourier transform
    HOP_LENGTH=512 #Hop lenght in samples for the mfcss
    NUM_SEGMENTS=5
    SAMPLES_PER_SEGMENT=int(SAMPLE_RATE*SAMPLE_TIME/NUM_SEGMENTS)
    NUM_MFCC_PER_SEG=math.ceil(SAMPLES_PER_SEGMENT/HOP_LENGTH)
    RANDOM_STATE_=42
    
    #SVM:
    #Original search
    #kernel_grid_search=['poly','linear','rbf']
    #c_grid_search=[0.01,1,10,100,1000]
    #coef0_grid_search=[2,3,6,12,18,24]
    #degree_grid_search=[3,4,5,6]

    #Best performance:
    kernel_grid_search=['poly']
    c_grid_search=[10]
    coef0_grid_search=[3]
    degree_grid_search=[3]

    #Random Forests
    #Original search
    #m_depth=[10,50,100,None]
    #n_estimators=[100,1000,1000]

    #Best performance:
    m_depth=[10]
    n_estimators=[1000]
    
    m=MODEL(SAMPLE_TIME,SAMPLE_RATE,DIR_MAIN,
                 N_MFCC,N_FFT,HOP_LENGTH,NUM_SEGMENTS,
                 RANDOM_STATE_,kernel_grid_search,
                 c_grid_search,coef0_grid_search,degree_grid_search)
    m.get_mfccs()
    
    svm_vib=m.train_svm(source='vib')
    svm_audio=m.train_svm(source='audio')
    
    rf_vib=m.train_random_forests(source='vib')
    rf_audio=m.train_random_forests(source='audio')
    
    
    joblib.dump(svm_vib, "SVM_vibration.joblib")
    joblib.dump(svm_audio, "SVM_audio.joblib")
    joblib.dump(rf_vib, "rf_vibration.joblib")
    joblib.dump(rf_audio, "rf_audio.joblib")
    
    
    