import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('pickle_my_model', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('prashant PGI18CS030 - Classification Dataset2.csv')
# Extracting independent variable:
X = dataset.iloc[:,0:-1].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age):
  output= model.predict(sc.transform([[Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("Disease", output)
  if output==[1]:
    prediction="You have a disease diagnostic"
  else:
    prediction="You have a disease diagnostic"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Red;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Disease DiagnosticPrediction using Logistic Classification</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Gender = st.number_input('Insert Gender Male:1 Female:0')
    Glucose=st.number_input('Insert a Glucose',50,200)
    BP=st.number_input('Insert a BP',50,200)
    SkinThickness=st.number_input('Insert a SkinThickness',50,100)
    Insulin=st.number_input('Insert a Insulin',50,100)
    BMI=st.number_input('Insert a BMI',0,100)
    PedigreeFunction=st.number_input('Insert a PedigreeFunction',0,100)
    Age = st.number_input('Insert a Age',18,60)
  
    result=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,BP,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Prashant Jain")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()
