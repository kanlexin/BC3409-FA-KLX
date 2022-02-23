#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        
        lr_model = joblib.load("LR")
        cart_model = joblib.load("CART")
        rf_model = joblib.load("RF")
        xgb_model = joblib.load("XGB")
        nn_model = joblib.load("NNMLP")
        
        lr_pred = lr_model.predict([[float(income), float(age), float(loan)]])
        cart_pred = cart_model.predict([[float(income), float(age), float(loan)]])
        rf_pred = rf_model.predict([[float(income), float(age), float(loan)]])
        xgb_pred = xgb_model.predict([[float(income), float(age), float(loan)]])
        nn_pred = nn_model.predict([[float(income), float(age), float(loan)]])
        
        print(lr_pred)
        print(cart_pred)
        print(rf_pred)
        print(xgb_pred)
        print(nn_pred)
        
#         lr_pred = lr_pred[0]
#         cart_pred = cart_pred[0]
#         rf_pred = rf_pred[0]
#         xgb_pred = xgb_pred[0]
#         nn_pred = nn_pred[0]
        
        lr = "The predicted default on credit card is " + str(lr_pred)
        cart = "The predicted default on credit card is " + str(cart_pred)
        rf = "The predicted default on credit card is " + str(rf_pred)
        xgb = "The predicted default on credit card is " + str(xgb_pred)
        nn = "The predicted default on credit card is " + str(nn_pred)
        
        return(render_template("index.html", result1 = lr, result2 = cart, result3 = rf, result4 = xgb, result5 = nn))
    else: 
        return(render_template("index.html", result1 = "", result2 = "", result3 = "", result4 = "", result5 = ""))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




