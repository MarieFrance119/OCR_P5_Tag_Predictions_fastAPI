#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:07:27 2022

@author: MFLB
"""
# Contient l'ensemble des fonctions nécessaires
# pour l'exécution du main_code

# Manipulation des données
import numpy as np
import pandas as pd

# Librairie nltk pour traiter les mots
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vectorisation des mots
from sklearn.feature_extraction.text import TfidfVectorizer

# Outils de sklearn
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Réduction de dimension
from sklearn.decomposition import TruncatedSVD

# Modèles
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

# Data Persistence
import joblib

# Pré-traitement du texte
def get_preprocessed_text(raw_question,raw_text):
    
    """
    Fonction pour traiter une question et un post
    en une liste de mots utilisable pour la modélisation
    
    - Arguments :
        - raw_question : question d'origine
        - raw_text : texte d'origine
    
    - Retourne :
        - text : le texte pré-traité, prêt pour la modélisation
    """
    
    # fusion de la question et du post
    text = raw_question + " " + raw_text
    
    # Tokenizer pour récupérer que les termes avec des lettres
    tokenizer = nltk.RegexpTokenizer(r"[a-zA-Z]+")

    # Récupération de la liste des mots mis en minuscule
    tokens = tokenizer.tokenize(text.lower())

    # On ne garde que les mots d'au moins 3 lettres
    tokens = list(filter(lambda x: len(x) >= 3, tokens))

    # Récupération des stopwords English
    sw = set(stopwords.words("english"))

    # Supprimer les stop words
    token_cleaned = [token for token in tokens if not token in sw]
    
    # POS tagging
    pos_t = nltk.pos_tag(token_cleaned)

    # Récupération des noms communs
    token_noun = [token[0] for token in pos_t if token[1] in ("NN", "NNS")]
    
    # Initialisation du lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Récupération des lems
    text = [lemmatizer.lemmatize(token) for token in token_noun]
    
    return text

# import du MultilabelBinarizer
filename_mlb = "./models/mlb_model.joblib"
mlb = joblib.load(filename_mlb)

# prédiction des tags
def predict_tags(model, lemmed_text):

    """ Fonction pour afficher la liste des tags prédits par le
    modèle
    
    - Arguments :
        - model : modèle à utiliser pour la prédiction
        - lemmed_text : texte dont il faut prédire les tags
    
    - Retourne:
         - pred_tags_list : la liste des tags, en filtrant sur les mots 
         présents dans le texte
    """

    # Application du modèle
    text_model = model.predict(lemmed_text)

    # Récupération des labels
    text_tags = mlb.inverse_transform(text_model)

    # Liste des tags
    pred_tags_list = list(
        {tag for tag_list in text_tags for tag in tag_list if (len(tag_list) != 0)}
    )

    # Filtrage sur les mots présents dans le lemmed_text
    pred_tags_list = [tag for tag in pred_tags_list if tag in lemmed_text]

    return pred_tags_list