from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi import Query, Path
from pydantic import field_validator, model_validator
from typing import Annotated, Optional
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from pipeline.pipeline_module import create_pipeline

app = FastAPI()

@app.get('/intro')
def intro():
    return {"Message":"Hey this is just a introduction page for stock market price prediction"}

@app.post('/get-ticker/{ticker}')
def get_ticker(ticker: str = Path(..., description="This is just for an example")):
    return {"Hello":"Testing Purpose"}
    
    

    