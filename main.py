from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi import Query, Path
from pydantic import field_validator, model_validator
from typing import Annotated, Optional
import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

app = FastAPI()

@app.get('/intro')
def intro():
    return {"Message":"Hey this is just a introduction page for stock market price prediction"}

@app.post('/get-ticker/{ticker}')
def get_ticker(ticker: str = Path(..., description="This is just for an example")):
    curr_date = datetime.date.today()
    data = yf.download(ticker, start = curr_date - relativedelta(years=5), end = curr_date, auto_adjust=True)
    if data is not None:
        df = data.to_csv('data', index=False)
        df = pd.read_csv('data')
        return df.head(5).to_json()
    else:
        raise TypeError('The collected data is None')

    
    

    