from prepare_dataset import *

import shap
import lime
from lime import lime_tabular

import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def lime_perturbed_pred(coef,X,exp):
    
    y_surrogate = (np.dot(X,np.asarray(coef).reshape(-1,1))+exp.intercept[1])
    #print(y_surrogate)
    y_surrogate = [0 if x <0.5 else 1 for x in y_surrogate]   
    
    return y_surrogate

#how similar is performance of original model and surrogate model on perturbed data
def fidelity_lime(lime_pred,lime_perturbed_y):
    print(accuracy_score(lime_pred,lime_perturbed_y))
    return accuracy_score(lime_pred,lime_perturbed_y)


def load_dataset():
	dataset_name = 'adult.csv'
	path_data = './datasets/'
	dataset = prepare_adult_dataset(dataset_name, path_data)
	X, y = dataset['X'], dataset['y']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	class_name = dataset['class_name']
	columns = dataset['columns']
	continuous = dataset['continuous']
	possible_outcomes = dataset['possible_outcomes']
	label_encoder = dataset['label_encoder']


	feature_names = list(columns)
	feature_names.remove(class_name)

	categorical_names = dict()
	idx_discrete_features = list()
	for idx, col in enumerate(feature_names):
		if col == class_name or col in continuous:
			continue
		idx_discrete_features.append(idx)
		categorical_names[idx] = label_encoder[col].classes_
	return X_train,y_train,X_test,y_test,feature_names,categorical_names



def build_model(X_train,y_train):

	rf = RandomForestClassifier(n_estimators=100,random_state=0).fit(X_train,y_train)
	return rf
## main for lime

def lime_run(rf,X_train,y_train,X_test,y_test,i,feature_names):

	lm=[]

	explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=range((X_train.shape[1])), class_names=[0,1], discretize_continuous=False,verbose=True,sample_around_instance=True)
	n_ft = len(feature_names)
	
	
	#generate a random point and test using UI
	
	values = st.slider('Select number of sampling points',     200, 2000,500)
	exp = explainer.explain_instance(X_test[i], rf.predict_proba, num_features=n_ft,num_samples=values)
	coef = [0]*n_ft
	for i in exp.as_list():
		coef[i[0]] = i[1]
	X_lime_scaled = exp.scaled_data
	X_lime = exp.scaled_data*explainer.scaler.scale_ + explainer.scaler.mean_
	y_lime = rf.predict(X_lime)
	plt.bar(feature_names,coef)
	plt.xticks(rotation=45)
	fig_size = plt.gcf().get_size_inches() #Get current size
	sizefactor = 2 #Set a zoom factor
	# Modify the current size by the factor
	plt.gcf().set_size_inches(sizefactor * fig_size)
	st.write(plt.gcf())
	lime_pred = lime_perturbed_pred(coef,X_lime_scaled,exp)
	sn = fidelity_lime(lime_pred,y_lime)
	lm.append(sn)
	st.write(sn)
	plt.close()

def shap_run(rf,X_train,y_train,X_test,y_test,i,feature_names):

	explainer = shap.TreeExplainer(rf,feature_perturbation='interventional', check_additivity=False)
	shap_values = explainer.shap_values(X_test[i].reshape(1,12))
	st.write("Base value ",explainer.expected_value[0])
	plt.bar(feature_names,list(shap_values[0].reshape(-1)))
	plt.xticks(rotation=45)
	fig_size = plt.gcf().get_size_inches() #Get current size
	sizefactor = 2 #Set a zoom factor
	plt.gcf().set_size_inches(sizefactor * fig_size)
	st.write(plt.gcf())
	plt.close()

if __name__=='__main__':
	
	st.title("The explanation exploration")
	

	X_train,y_train,X_test,y_test,feature_names,categorical_names = load_dataset()
	test_index = int(st.text_input('Please give index of test point between 0 to '+str(X_test.shape[0]),10))

	st.write(pd.DataFrame(X_test[test_index].reshape(1,-1),columns=feature_names))

	model = build_model(X_train,y_train)
	st.write(model.predict_proba(X_test[test_index].reshape(1,-1)))

	if st.checkbox("LIME"):
		lime_run(model,X_train,y_train,X_test,y_test,test_index,feature_names)
	if st.checkbox("SHAP"):
		shap_run(model,X_train,y_train,X_test,y_test,test_index,feature_names)





