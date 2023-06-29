from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI(title="API startup",
              description="with FastAPI", version="1.0")

# Basemodel


class CompanyData(BaseModel):
    rd: float = 73721
    admin: float = 121344
    market: float = 211025

# blocco per la cache del mio modello


@app.on_event("startup")
def startup_event():

    global model
    model = joblib.load("model.pkl")
    print(" MODEL LOADED!!")
    return model

##########################################################################################################
################################# GET POST ##############################################


@app.get("/")
def home():
    return {" ---->          http://localhost:8001/docs     <----------"}


@app.get("/predict")
async def predictget(data: CompanyData = Depends()):
    try:
        X = [[data.rd, data.admin, data.market]]
        y_pred = model.predict(X)[0]
        res = round(y_pred, 2)
        return {'prediction': res}
    except:
        raise HTTPException(status_code=404, detail="error")


@app.post("/predict")
async def predictpost(data: CompanyData):
    try:
        X = [[data.rd, data.admin, data.market]]
        y_pred = model.predict(X)[0]
        res = round(y_pred, 2)
        return {'prediction': res}
    except:
        raise HTTPException(status_code=404, detail="error")

###############################################################################################

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
