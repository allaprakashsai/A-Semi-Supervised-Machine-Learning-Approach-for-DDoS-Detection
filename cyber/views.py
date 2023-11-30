from django.shortcuts import render
from django.contrib import messages
from user.models import Usermodel
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def index(request):
    return render(request, "index.html")

def Home(request):
    return index(request)

def adminlogin(request):
    return render(request, "admin/adminlogin.html")

def adminloginaction(request):
    if request.method == 'POST':
        uname = request.POST['uname']
        passwd = request.POST['upasswd']
        if uname == 'admin' and passwd == 'admin':
            data = Usermodel.objects.all()
            return render(request, "admin/adminhome.html", {'data': data})
        else:
            messages.success(request, 'Incorrect Details')
            return render(request, "admin/adminlogin.html")
    return render(request, "admin/adminlogin.html")

def showusers(request):
    data = Usermodel.objects.all()
    return render(request, "admin/adminhome.html", {'data': data})

def AdminActiveUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        Usermodel.objects.filter(id=id).update(status=status)
        data = Usermodel.objects.all()
        return render(request, "admin/adminhome.html", {'data': data})

def logout(request):
    return render(request, "admin/adminlogin.html")

def Ml(request):

    df=pd.read_csv(os.path.join(BASE_DIR, 'media/data.csv'),low_memory=False)

    data=df.to_numpy()
    data.shape

    #  Selecting the Features
    X = df[[' Fwd Packet Length Min', ' Fwd Packet Length Std', ' Flow IAT Mean',' Flow IAT Max', ' Fwd IAT Max', ' Bwd IAT Mean', ' Bwd IAT Min',' RST Flag Count', ' Avg Bwd Segment Size', ' Init_Win_bytes_backward']]
    y = df[' Label']

    print(X.shape)
    print(y.shape)

    # create 10 folds for cross validation
    folds=StratifiedKFold(n_splits=10)
    folds

    # split data to train and test parts
    x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.3)

    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix)
    import numpy as np

    def plot_confusion_matrix(confusion_matrix, classes):
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

    def output(model, x_train, x_test, y_train, y_test, folds):
        y_pred = cross_val_predict(model, x_train, y_train, cv=folds)
        accuracy = accuracy_score(y_pred, y_train) * 100
        precision = precision_score(y_train, y_pred) * 100
        recall = recall_score(y_train, y_pred) * 100
        f1 = f1_score(y_train, y_pred) * 100
        matrix = confusion_matrix(y_train, y_pred)
        plot_confusion_matrix(matrix, classes=['0', '1'])
        plt.show()
        return accuracy, precision, recall, f1


    model_lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    output_lr = output(model_lr, x_train, x_test, y_train, y_test, folds)

    # random forest model with 5 estimators
    model_rf = RandomForestClassifier(n_estimators=5)
    output_rf = output(model_rf, x_train, x_test, y_train, y_test, folds)

    # decision tree model
    model_dt = tree.DecisionTreeClassifier()
    output_dt = output(model_dt, x_train, x_test, y_train, y_test, folds)

    # gaussian naive bayes model
    model_nb = GaussianNB()
    output_nb = output(model_nb, x_train, x_test, y_train, y_test, folds)

    # KNN model with 1 neighbor
    model_knn = KNeighborsClassifier(n_neighbors=1)
    output_knn = output(model_knn, x_train, x_test, y_train, y_test, folds)
    
    return render(request, "admin/Ml.html",{
            'output_lr': output_lr,
            'output_rf': output_rf,
            'output_dt': output_dt,
            'output_nb': output_nb,
            'output_knn': output_knn,
        })