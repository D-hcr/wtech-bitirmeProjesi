from fastapi import FastAPI
from pydantic import BaseModel 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import load_model 

app = FastAPI()

"""data = {
    "Gender": 1,
    "Country": 34,
    "Occupation": 2,
    "self_employed": 1,
    "family_history": 0,
    "treatment": 0,
    "Days_Indoors": 0,
    "Growing_Stress": 1,
    "Changes_Habits": 2,
    "Mental_Health_History": 1,
    "Coping_Struggles": 1,
    "Work_Interest": 1,
    "Social_Weakness": 2,
    "mental_health_interview": 1,
    "care_options": 2
}"""

   
class MentalHealthModel(BaseModel):
    Gender:int
    Country:int
    Occupation:int
    self_employed:int
    family_history:int
    treatment:int
    Days_Indoors:int
    Growing_Stress:int
    Changes_Habits:int
    Mental_Health_History:int
    Coping_Struggles:int
    Work_Interest:int
    Social_Weakness:int
    mental_health_interview:int
    care_options:int

@app.post("/predict_mentalHealth/")
async def predict_mentalHealth(item: MentalHealthModel):
    load_ann = load_model('bitirmeProjesi.h5')
    
    sc = StandardScaler()
    data = pd.DataFrame([item.dict()])
   
    sc.fit(data)
    data = sc.transform(data)
    prediction = load_ann.predict(data)
    
    prediction = prediction[0]
    
    health = prediction.argmax()
    
    if health == 0:
        health = "LOW"
    elif health == 1:
        health = "HIGH"
    else:
        health = "MEDIUM"
    return {"health": health}
