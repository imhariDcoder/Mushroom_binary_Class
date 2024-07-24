import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import (precision_recall_curve,PrecisionRecallDisplay)
import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrics,plot_ruc_curve,plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score, confusion_matrix


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Mushy Pussy App")
    st.markdown("Are your mushrooms edible or poisionous ? 🍄 ")
    st.sidebar.markdown("Is your pussy edible ? 🍄 ")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=["type"])
        x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
        return x_train,x_test,y_train,y_test
    
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, model.predict(x_test))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot()
            st.pyplot(plt)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")  
            # Assuming you're using a binary classifier
            # y_proba = model.predict_proba(x_test)[:, 1]  # Probability of the positive class
            disp = RocCurveDisplay.from_estimator(model, x_test, y_test)  
            disp.plot()
            st.pyplot(plt)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            disp = PrecisionRecallDisplay(precision=precision,recall=recall)
            disp.plot()
            st.pyplot(plt)
    

 



    df = load_data()
    x_train,x_test,y_train,y_test = split(df)
    class_names = ("edible", "poisionous")
    st.sidebar.subheader("Chose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("SVM","Logistic Regression","Random Forest Classifier"))

    if classifier == 'SVM':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)",0.01, 10.0,step=0.01,key='C')
        kernel = st.sidebar.radio('Kernel',('rbf','linear'),key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel coefficient)",('scale','auto'),key='Gamma')

        metrics = st.sidebar.multiselect("What metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))


        if st.sidebar.button("Classify",key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C,kernel = kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",round(accuracy,2))
            st.write("Precision: ",round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write("Recall: ",round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics,model,x_test,y_test,class_names)

        
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C = st.sidebar.number_input("C (Regularization parameter)",0.01, 10.0,step=0.01,key='C_LR')
        max_itr = st.sidebar.slider("Maximum number of iteration",100,500,key = 'max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))


        if st.sidebar.button("Classify",key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C,max_iter=max_itr)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",round(accuracy,2))
            st.write("Precision: ",round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write("Recall: ",round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics,model,x_test,y_test,class_names)

    if classifier == 'Random Forest Classifier':
        st.sidebar.subheader('Model Hyperparameters')
        n_estimators = st.sidebar.number_input("The number of tree in the forest",100,5000,step=10,key='RF_est')
        max_depth = st.sidebar.number_input("The max depth of the tree",1,20,step = 1,key='RF_Maxdepth')
        bootstrap = st.sidebar.radio("Bootstrap sample while building tree",(True,False),key='Bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot ? ",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))


        if st.sidebar.button("Classify",key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",round(accuracy,2))
            st.write("Precision: ",round(precision_score(y_test,y_pred,labels=class_names),2))
            st.write("Recall: ",round(recall_score(y_test,y_pred,labels=class_names),2))
            plot_metrics(metrics,model,x_test,y_test,class_names)



    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)







if __name__=='__main__':
    main()