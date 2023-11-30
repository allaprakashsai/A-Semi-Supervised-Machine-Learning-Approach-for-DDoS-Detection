from django.shortcuts import render
from django.contrib import messages
from user.models import Usermodel
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Create your views here.
def Userlogin(request):
    return render(request, "user/Userlogin.html")

def userregister(request):
    return render(request, "user/userregister.html")

def userregisterAction(request):
    if request.method == 'POST':
        name = request.POST.get('uname')
        email = request.POST.get('uemail')
        password = request.POST.get('upasswd')
        phoneno = request.POST.get('uphonenumber')
        form1 = Usermodel(name=name, email=email, password=password, phoneno=phoneno, status='waiting')
        form1.save()
        messages.success(request, 'Registration Successful')
        return render(request, "user/Userlogin.html")
    else:
        messages.error(request, 'Registration Unsuccessful')
        return render(request, "user/userregister.html")

def userloginaction(request):
    if request.method == 'POST':
        sname = request.POST.get('email')
        spasswd = request.POST.get('upasswd')
        try:
            check = Usermodel.objects.get(email=sname, password=spasswd)
            if check.status == 'activated':
                messages.success(request, 'Login Successful')
                return render(request, "user/userhome.html")
        except Usermodel.DoesNotExist:
            pass
        messages.error(request, 'Login Unsuccessful')
    return render(request, "user/Userlogin.html")

    
def usrhome(request):
    return render(request, "user/userhome.html")

def predict(request):
    if request.method == 'POST':
        l1 = float(request.POST.get('input1'))
        l2 = float(request.POST.get('input2'))
        l3 = float(request.POST.get('input3'))
        l4 = float(request.POST.get('input4'))
        l5 = float(request.POST.get('input5'))
        l6 = float(request.POST.get('input6'))
        l7 = float(request.POST.get('input7'))
        l8 = float(request.POST.get('input8'))
        l9 = float(request.POST.get('input9'))
        l10 = float(request.POST.get('input10'))
        print(l1,l2,l3,l4,l5,l6,l7,l8,l9, l10)

        # reading the data
        df=pd.read_csv(os.path.join(BASE_DIR, 'media/data.csv'),low_memory=False)

        #  converting data to linear algebra
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
        x_train, x_test, y_train, y_test=train_test_split(X,y,train_size=0.3)

        # Working with model
        model = LogisticRegression()
        model.fit(x_train, y_train)

        def plot_confusion_matrix(confusion_matrix, classes):
            plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

        def output(model, x_train, x_test, y_train, y_test, folds, input_sample):
            y_pred = cross_val_predict(model, x_train, y_train, cv=folds)
            print(f"Accuracy_score: {accuracy_score(y_pred, y_train)*100}")
            print(f"precision_score: {precision_score(y_train, y_pred)*100}")
            print(f"recall_score: {recall_score(y_train, y_pred)*100}")
            print(f"f1_score: {f1_score(y_train, y_pred)*100}")
            matrix = confusion_matrix(y_train, y_pred)
            plot_confusion_matrix(matrix, classes=['0', '1'])  # Replace with your class labels
            plt.show()

            # Predict on input sample
            global input_pred
            input_pred = model.predict(input_sample)
            print(f"Input prediction: {input_pred}")
            if input_pred == 1:
                print('DDoS attack') # 1 represents for DDoS attack
            else:
                print('No DDoS attack') # 0 represent for no attack
        input_sample = [[l1,l2,l3,l4,l5,l6,l7,l8,l9, l10]]
        print(input_sample)
        output(model, x_train, x_test, y_train, y_test, folds, input_sample)
        return render(request, "user/output.html", {'pred':input_pred})
    else:
        return render(request, "user/userhome.html")

def usrlogout(request):
    return render(request, "user/userlogin.html")