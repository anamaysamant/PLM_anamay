#Multi/Binary Classification
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from keras.layers import GaussianNoise
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import save_model
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from itertools import cycle
#For Balancing and Scaling: 
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

#This Classification function is used to Classify output embeddings from BERT models. 
#Note: Classification can be Binary of Multi-Class depending on feature labels
#@param embeddings takes a pd.DataFrame of embeddings as input.
#@param labels takes a list of feature labels which matches the rows of the embeddings describing the data embedded. 
#Note: embeddings and labels length must equal rows of embeddings. 
#@param 'eval_size' Default is 0.1, input must be between 0-1. 
#The eval size is the percentage of data used to Evaluate the models performance, 0.1 = 10% of the embeddings. 
#@param 'balancing' takes a boolean as input: default is True.
#Balacing does RandomOverSampling of all minority classes so that the number of minority equals majority class.
#@param 'epochs' takes a numeric input: Default is 3 times the number of embedding columns
#Epochs dictates how many passes the model will undergo during configuration, used to update weight, bias, and parameter optimization.
#@param 'nodes' takes a numeric input: Default is 1/2 the number of embedding columns.
#Influences how many nodes will consist in the single layered neural network. 
#@param 'batch_size' takes a numeric value: Default is 32.
#the batch size determines how many training samples are processed together. 
#@param 'scaling' takes boolean input:
#Uses the sklearn StandardScaler function to scale data
#@param 'display' takes four inputs ('both', 'none', 'cm', or 'roc'), default is 'none':
#'display' = 'both', makes confusion matrices and ROC curves for all models.
#@param 'patience', takes integer input greater than 0: Default is 15.
#Determines how many epochs the model will train for with no decreases in loss.
#Note: The larger the patience value the larger the chance of the neural network overfitting.
#@param 'noise', takes boolean input, default is False:
#Creates an initial Gaussian noise layer before hidden layers of neural network. 
#@param 'return_model', takes boolean input, default is False:
#Returns the neural network made from the get_Classification function.
#Note: if 'True' need to specify an output variable for the model. 

# @EXAMPLE:
#import Classification.py as clf
# embeddings = pd.read_csv('embeddings.csv')
# metadata = pd.read_csv('metadata')
# labels = metadata['labels']
# clf.get_Classification(embeddings, labels) 
# or 
# clf.get_Classification(embeddings, labels, balancing=False, eval_size=0.2, epochs = 200, nodes = 100, batch_size= 16, scaling=True, display = True, patience = 20)


def get_Classification(embeddings, labels, balancing=False, eval_size=0.1, epochs=0, nodes=0, batch_size=32, scaling=False, display='none', patience=15, noise = False, return_model=False):
    #Default epochs is three times the number of columns 
    if(epochs == 0): 
        epochs = len(embeddings.columns)*3
    #Default nodes is one half the number of columns 
    if(nodes == 0):
        nodes = int(len(embeddings.columns)/2)
    plt_labels = list(np.unique(labels))
    #Make Model and Subset Evaluation Data
    model, X_eval, Y_eval = make_NN(embeddings, labels, balancing = balancing, eval_size = eval_size, epochs = epochs,
                                                nodes = nodes, batch_size = batch_size, scale = scaling, patience = patience, noise = noise)
    Y_eval = np.array(Y_eval)
    #Get probabilites from NN
    Y_pred = model.predict(X_eval)
    if(len(plt_labels) == 2):
        print('NN Report', classification_report(Y_eval, np.round(Y_pred).astype(int), target_names = plt_labels))
    else:
        print('NN Report', classification_report(Y_eval.argmax(axis = 1), Y_pred.argmax(axis=1), target_names = plt_labels))
    
    #Make SVM and probabilites
    print('Making SVM Model')
    Y_test, SVM_pred = make_model('SVM', embeddings, labels, eval_size, balancing, scaling)
    print('SVM Report',classification_report(Y_test, prob2pred(SVM_pred, len(plt_labels)), target_names = plt_labels))
    print('Making Random Forest Model')
    RF_pred = make_model('RF', embeddings, labels, eval_size, balancing, scaling)
    print('RF Report',classification_report(Y_test, prob2pred(RF_pred, len(plt_labels)), target_names = plt_labels))
    print('Making Gaussian Naive Bayes Model')
    GNB_pred = make_model('GNB', embeddings, labels, eval_size, balancing, scaling)
    print('GNB Report', classification_report(Y_test, prob2pred(GNB_pred, len(plt_labels)), target_names = plt_labels))
    print('Making Logistic Regression Model') 
    LGR_pred = make_model('LOGREG', embeddings, labels, eval_size, balancing, scaling)
    print('LGR Report', classification_report(Y_test, prob2pred(LGR_pred, len(plt_labels)), target_names = plt_labels))
    if(display != 'none'):
        #NN Confusion Matrices: 
        if display == 'both' or display == 'cm' :
            if(len(plt_labels) == 2):
                nn_cm = get_CM(Y_eval, np.round(Y_pred.max(axis=1)).astype(int), plt_labels)
            else:
                nn_cm = get_CM(Y_eval.argmax(axis=1), Y_pred.argmax(axis=1), plt_labels)
            #SVM Confusion Matrices: 
            svm_cm = get_CM(Y_test, prob2pred(SVM_pred, len(plt_labels)), plt_labels)
            #RF Confusion Matrices:
            rf_cm = get_CM(Y_test, prob2pred(RF_pred, len(plt_labels)), plt_labels)
            #GNB Confusion Matrix
            gnb_cm = get_CM(Y_test, prob2pred(GNB_pred, len(plt_labels)), plt_labels)
            #LGR Confusion Matrix
            lgr_cm = get_CM(Y_test, prob2pred(LGR_pred, len(plt_labels)), plt_labels)
            #Plot Confusion Matrices
            plot_cms([nn_cm, svm_cm, rf_cm, gnb_cm, lgr_cm], plt_labels)
            #Plot ROCs 
        if display == 'both' or display == 'roc' :
            if(len(plt_labels)==2):
                make_ROC(Y_eval, [Y_pred,SVM_pred[:,1], RF_pred[:,1], GNB_pred[:,1], LGR_pred[:,1]], plt_labels)
            else: 
                make_ROC(Y_eval, [Y_pred,SVM_pred, RF_pred, GNB_pred, LGR_pred], plt_labels)
    if return_model == True : 
        print('Returning Neural Network')
        return model

#Coverts probabilites to predictions 
def prob2pred(Y_prob, count):
    if(count == 2):
        return np.round(Y_prob[:,1]).astype(int)
    else:
        return Y_prob.argmax(axis=1)

#Input Embeddings (DF) and labels (Array)
def make_NN(embeddings, labels, balancing, eval_size, epochs, nodes, batch_size, scale, patience, noise):
    X = np.array(embeddings)
    #Encode labels: 
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    if(len(set(labels))>2):
        Y = np_utils.to_categorical(Y)
    X = X.astype(float)
    #Split evaluation sets:
    X, X_eval, Y, Y_eval = train_test_split(X, Y, test_size = eval_size, random_state = 42)
    #Optional Balancing: Random Over Sampling of all classes but highest
    if(balancing == True):
            oversample = RandomOverSampler(sampling_strategy='not majority')
            X, Y = oversample.fit_resample(X, Y)
    #Split training and test sets: 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    #Optional Scale: Sklearn StandardScaler
    if(scale == True):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_eval = scaler.transform(X_eval)
    #Run Model:
    #If Binary Classification:
    if(len(set(labels))==2):
        print("Running Binary Classification")
        callback = EarlyStopping(monitor='val_loss', patience=patience, mode = 'min', verbose = 1, restore_best_weights=True)
        model = Sequential()
        if(noise==True):
            model.add(GaussianNoise(1))
        model.add(Dense(nodes, input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size = batch_size, verbose = 1,
                    epochs = epochs, validation_data = (X_test, Y_test), callbacks=[callback], shuffle = False)
        _, accuracy = model.evaluate(X_test,Y_test)
        print("Neural Network Evaluation Accuracy = ", (accuracy*100),"%")
        return model, X_eval, Y_eval
     #If Multi-class Classification:
    elif(len(set(labels))>2):
        print("Running Multi-class Classification")
        callback = EarlyStopping(monitor='val_loss', patience=patience, mode = 'min', verbose = 1, min_delta = .001, restore_best_weights=True)
        model = Sequential()
        if(noise == True):
            model.add(GaussianNoise(1))
        model.add(Dense(nodes, input_dim=(len(embeddings.columns)), activation='relu'))
        model.add(Dense(len(set(labels)), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size = batch_size, verbose = 1,
                    epochs = epochs, validation_data = (X_test, Y_test), callbacks=[callback], shuffle = False)
        _, accuracy = model.evaluate(X_eval,Y_eval)
        print("Neural Network Evaluation Accuracy = ", (accuracy*100),"%")
        return model, X_eval, Y_eval
    
def get_CM(Y_act, Y_pred, labels):
    cm = confusion_matrix(Y_act, Y_pred, labels = np.unique(Y_act))
    cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels)
    return cm

def make_ROC(Y_test, prob_list, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Calculate Binary ROCs
    if(len(labels)==2):
        fpr, tpr, roc_auc_model = [], [], []
        for i in range(len(prob_list)):
            fprt, tprt, thresholdst = roc_curve(Y_test, prob_list[i])
            auct = auc(fprt, tprt)
            fpr.append(fprt)
            tpr.append(tprt)
            roc_auc_model.append(auct)
        plt.figure()
        plt.plot(fpr[0], tpr[0], color='darkorange', lw=2, label=f'Neural Network (AUC = {roc_auc_model[0]:.2f})')
        plt.plot(fpr[1], tpr[1], color='green', lw=2, label=f'SVM (AUC = {roc_auc_model[1]:.2f})')
        plt.plot(fpr[2], tpr[2], color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_model[2]:.2f})')
        plt.plot(fpr[3], tpr[3], color='red', lw=2, label=f'Gaussian NB (AUC = {roc_auc_model[3]:.2f})')
        plt.plot(fpr[4], tpr[4], color='purple', lw=2, label=f'Logistic Regression (AUC = {roc_auc_model[4]:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
    elif(len(labels)>2):
        #Calculate Micro-Averages: 
        fpr_mi, tpr_mi, auc_mi = [], [], []
        for i in range(len(prob_list)):
            fpr_mi_temp, tpr_mi_temp, _ = roc_curve(Y_test.ravel(), prob_list[i].ravel())
            auc_mi_temp = auc(fpr_mi_temp, tpr_mi_temp)
            fpr_mi.append(fpr_mi_temp)
            tpr_mi.append(tpr_mi_temp)
            auc_mi.append(auc_mi_temp)
        plt.figure()
        plt.plot(fpr_mi[0], tpr_mi[0], color='darkorange', lw=2, label=f'Neural Network (AUC = {auc_mi[0]:.2f})')
        plt.plot(fpr_mi[1], tpr_mi[1], color='green', lw=2, label=f'SVM (AUC = {auc_mi[1]:.2f})')
        plt.plot(fpr_mi[2], tpr_mi[2], color='blue', lw=2, label=f'Random Forest (AUC = {auc_mi[2]:.2f})')
        plt.plot(fpr_mi[3], tpr_mi[3], color='red', lw=2, label=f'Gaussian NB (AUC = {auc_mi[3]:.2f})')
        plt.plot(fpr_mi[4], tpr_mi[4], color='purple', lw=2, label=f'Logistic Regression (AUC = {auc_mi[4]:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-Average Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()
        #Calculate Macro-Averages
        n_classes = len(Y_test[1])
        fpr_ma, tpr_ma, auc_ma = [], [], []
        for n in range(len(prob_list)):
            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prob_list[n][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            mean_tpr = np.zeros_like(fpr_grid)
            for i in range(n_classes):
                mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i]) 
            #Compute AUC
            mean_tpr /= n_classes
            fpr_ma.append(fpr_grid)
            tpr_ma.append(mean_tpr)
            auc_ma.append(auc(fpr_ma[n], tpr_ma[n]))
        plt.figure()
        plt.plot(fpr_ma[0], tpr_ma[0], color='darkorange', lw=2, label=f'Neural Network (AUC = {auc_ma[0]:.2f})')
        plt.plot(fpr_ma[1], tpr_ma[1], color='green', lw=2, label=f'SVM (AUC = {auc_ma[1]:.2f})')
        plt.plot(fpr_ma[2], tpr_ma[2], color='blue', lw=2, label=f'Random Forest (AUC = {auc_ma[2]:.2f})')
        plt.plot(fpr_ma[3], tpr_ma[3], color='red', lw=2, label=f'Gaussian NB (AUC = {auc_ma[3]:.2f})')
        plt.plot(fpr_ma[4], tpr_ma[4], color='purple', lw=2, label=f'Logistic Regression (AUC = {auc_ma[4]:.2f})')
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-Average Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

def make_SVM(X_train, Y_train): 
    if(len(np.unique(Y_train)) > 2):
        model = SVC(decision_function_shape='ovo', probability = True)
        model.fit(X_train, Y_train)
        print(f'SVM Evaluation Accuracy - :{model.score(X_train, Y_train):.3f}')
        return model
    elif(len(np.unique(Y_train)) == 2):
        model = SVC(gamma = 'auto', probability = True)
        model.fit(X_train, Y_train)
        print(f'SVM Evaluation Accuracy - :{model.score(X_train, Y_train):.3f}')
        return model

def make_RF(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    print(f'Random Forest Evaluation Accuracy - :{model.score(X_train, Y_train):.3f}')
    return model

def make_GNB(X_train, Y_train):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    print(f'Gaussian Naive Bayes Evaluation Accuracy - :{model.score(X_train, Y_train):.3f}')
    return model

def make_LogReg(X_train, Y_train):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print(f'Logistic Regression Evaluation Accuracy - :{model.score(X_train, Y_train):.3f}')
    return model

def make_model(model, embeddings, labels, eval_size, balancing, scaling):
    X = np.array(embeddings)
    encoder = LabelEncoder()
    encoder.fit(labels)
    Y = encoder.transform(labels)
    X = X.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = eval_size, random_state = 42)
    #Optional Balancing:
    if(balancing == True):
        print('Balancing Data...')
        oversample = RandomOverSampler(sampling_strategy='not majority')
        X, Y = oversample.fit_resample(X, Y)
    #Optional Scaling:
    if(scaling == True):
        print('Scaling Data...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    #Make specificied models: 
    if(model == 'SVM'):
        model = make_SVM(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)
        print(f'SVM Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return Y_test, Y_pred
    elif(model == 'RF'):
        model = make_RF(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)
        print(f'Random Forest Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return Y_pred
    elif(model == 'GNB'):
        model = make_GNB(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)
        print(f'GNB Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return Y_pred
    elif(model == 'LOGREG'):
        model = make_LogReg(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)
        print(f' Logistic Regression Test Accuracy - :{model.score(X_test, Y_test):.3f}')
        return Y_pred

def plot_cms(cm_list, plt_labels):
    #For Visualization: 
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
    #Making subplot of all model Confusion Matrices: 
    titles = ['Neural Network','SVM', 'Random Forest', 'GaussianNB', 'Logistic Regression']
    for i, ax in enumerate(axes.flatten()):
        if i < len(cm_list):
            disp = cm_list[i]
            disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".0f")
            disp.ax_.set_title(titles[i])
            disp.im_.colorbar.remove()
            ax.set_xticklabels(plt_labels, rotation=45, ha="right")
        if i >= len(cm_list):
            ax.axis("off")
    plt.tight_layout()
    plt.show()
