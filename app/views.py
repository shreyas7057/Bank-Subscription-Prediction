from django.shortcuts import render,redirect
import pandas as pd
import numpy as np
import pickle


def getPredictions(age,job,marital,education,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,cons_price_idx,cons_conf_idx,euribor3m):
    model = pickle.load(open("diamond_price.pkl", "rb"))
    prediction = model.predict([[age,job,marital,education,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome,cons_price_idx,cons_conf_idx,euribor3m]])

    if prediction== 0:
        return "Customer Will not Subscribe For Scheme"
    
    elif prediction == 1:
        return "Customer Will Subscribe For Scheme"
    
    else:
        return "Error!!!"

def index(request):

    if request.method=='POST':
        carat = eval(request.POST.get('carat'))
        cut = eval(request.POST.get('cut'))
        color = eval(request.POST.get('color'))
        clarity = eval(request.POST.get('clarity'))
        depth = eval(request.POST.get('depth'))
        table = eval(request.POST.get('table'))
        x = eval(request.POST.get('x'))
        y = eval(request.POST.get('y'))
        z = eval(request.POST.get('z'))

        result = getPredictions()
        diamond_prices = np.around(result[0],2)

        context = {
            'result':diamond_prices,
        }
        return render(request,'index.html',context)