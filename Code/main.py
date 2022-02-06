#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 11:50:16 2022

@author: MFLB
"""


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from functions import get_preprocessed_text,get_tags,predict_tags

app = FastAPI(
    title='Tags Prediction Application',
    description='Application to give tags prediction for question with FastAPI + uvicorn',
    version='0.0.1')

@app.get("/")
def root():
    return {"Welcome to the API. Check /docs for usage"}

class Input(BaseModel):
    question : str
    description : str

@app.post("/predict")
async def get_prediction(data: Input):

    #Preprocessing du texte saisi
    preproc_text = get_preprocessed_text(data.question,data.description)
   
    # Envoi du texte dans le modèle pour avoir prédiction
    pred = predict_tags(preproc_text)
    
    # Transformation des tags en "mots"
    supervised_pred = get_tags(pred, preproc_text)
    
    # Encodage json
    question = jsonable_encoder(data.question)
    description = jsonable_encoder(data.description)

    return JSONResponse(status_code=200, 
                        content={"Enter your question": question,
                                 "Explain your problem" : description,
                                 "Tags prediction": supervised_pred})