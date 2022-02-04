#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:05:13 2022

@author: MFLB
"""

# Data Persistence
import joblib

# import des fonctions de traitement de textes
from functions import *


question_raw = "How to plot categorical variables with matplotlib python?"
text_raw = "How to plot categorical variables with matplotlib ?"

# Récupération du texte traité
preprocessed_text = get_preprocessed_text(question_raw,text_raw)
print(preprocessed_text)

# import de la liste de référence des mots 
filename_list_words = "./models/list_words.joblib"
list_words = joblib.load(filename_list_words)

# import du modèle
filename_pipeline_SVM_10 = "./models/pipeline_SVM_10.joblib"
pipeline_SVM_10 = joblib.load(filename_pipeline_SVM_10)

# Prédiction des tags
prediction = predict_tags(pipeline_SVM_10, preprocessed_text)

dico = {"tags" : prediction}
print(*prediction, sep = ', ')
print(dico.get("tags"))

