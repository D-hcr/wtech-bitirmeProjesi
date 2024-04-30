from fastapi import FastAPI
from pydantic import BaseModel # Fonksiyonlara gelen parametreleri kontrol etmek için
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # Modeli yükleme

# Define the FastAPI app
app = FastAPI()

data = {
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
}

@app.post("/predict_curn2/")
def predict_curn2(Gender:int, Country:int, Occupation:int, self_employed:int,
    family_history:int, treatment:int, Days_Indoors:int,
    Growing_Stress:int, Changes_Habits:int,
    Mental_Health_History:int, Coping_Struggles:int,
    Work_Interest:int, Social_Weakness:int,
    mental_health_interview:int, care_options:int):
    pass
    

# Deep Learning modelin ihtiyaç duyduğu inputları almak için bir class oluşturuyoruz.
# Model ayrıştırılıken X verisetinin kolonları ve türleri ile aynı olmalıdır.
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

@app.post("/predict_curn/") # url predict_curn
def predict(item: MentalHealthModel):
    load_ann = load_model('/home/hacer/Desktop/WtechBitirmeProjesi/bitirme_projesi.h5')

    # item dan gelen veriyi pandas dataframe çeviriyoruz
    # amacımız modelin beklentisine uygun bir veri yapısı oluşturmak
    data = pd.DataFrame([item.dict()])
    sc = StandardScaler()
    
    # verimizi scale ile fit ediyoruz
    sc.fit(data)
    # Transform your data
    data = sc.transform(data)
    print(data)

    # predict methodu ile modeli kullanarak tahmin yapılır
    prediction = load_ann.predict(data)
    print(prediction) # [[0.42079002]] iç içe liste döner
    print("----------------")
    mentalHealth = prediction[0][0]
    print(mentalHealth) # 0.42079002
    # Eğer mentalHealth değeri 0.5 den büyükse 1 küçükse 0 döndür
    if mentalHealth >= 0.5:
        mentalHealth = 1
    else:
        mentalHealth = 0

    return {"mentalHealth": mentalHealth}

