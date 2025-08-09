from fastapi import FastAPI, Path, HTTPException
from contextlib import asynccontextmanager
import sys
from threading import Thread

def load_pipeline_background(app_state):
    print('Background thread started: Loading prediction pipeline...')
    try:
        from pipeline.pipeline_module import create_prediction_pipeline, encode_year
        sys.modules["__main__"].encode_year = encode_year # type: ignore
        pipeline = create_prediction_pipeline() 
        app_state.pipe = pipeline 
        app_state.pipeline_loaded = True
        print('Prediction pipeline loaded successfully')
    except Exception as e:
        print(f'FATAL: Failed to load pipeline in background: {e}')
        app_state.pipe = None
        app_state.pipeline_loaded = False

@asynccontextmanager
async def lifespan(app:FastAPI):
    app.state.pipeline_loaded = False
    app.state.pipe = None

    loader_thread = Thread(target=load_pipeline_background, args=(app.state, ))
    loader_thread.start()
    print('Server started immediately. Pipeline loading in background')
    yield
    print('Shutting down')

app = FastAPI(lifespan=lifespan)

@app.get('/', tags=['General'])
def intro():
    return {'message':'This is Stock Price Prediction API'}

@app.get('/predict/{ticker}', tags=['Prediction'])
def get_predictions(ticker = Path(..., description="The Ticker Symbol of the stock (e.g., 'AAPL')")):
    if not app.state.pipeline_loaded or not app.state.pipe:
        raise HTTPException(status_code=503, detail="The prediction model is still being initialized. Please try again in a minute.")
    try:
        from pipeline.pipeline_module import gather_ticker_data
        new_data = gather_ticker_data(ticker_symbol=ticker)
        pipeline = app.state.pipe
        forecast = pipeline.predict(new_data, forecast_horizon = 7)
        response = {
            'ticker':ticker,
            'forecast':forecast.to_dict('records'),
            'historical':new_data.tail(30).reset_index().to_dict('records')
        }
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 