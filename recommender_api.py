import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# loading the dataset raw recipes
df = pd.read_csv("filtered_data.csv")

# Drop 'food types' column since it's not needed in processing
df.drop(['food types'], axis=1, inplace=True)

# Initialize the scaler and apply it to the appropriate columns
scaler = StandardScaler()
prep_data = scaler.fit_transform(df.iloc[:, 7:14].to_numpy())  # Adjust this range to the columns for nutritional values

# Initialize Nearest Neighbors model
neigh = NearestNeighbors(metric='cosine', algorithm='brute')
neigh.fit(prep_data)

# Function to scale data
def scaling(df):
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(df.iloc[:, 7:14].to_numpy())  # Adjust this to the relevant columns
    return prep_data, scaler

# Function to get nearest neighbors
def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric="cosine", algorithm="brute")
    neigh.fit(prep_data)
    return neigh

# Build pipeline
def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([("std_scaler", scaler), ("NN", transformer)])
    return pipeline

# Filter function to apply conditions on the data
def filter_data(df, ingredient_filter, max_nutrition, food_type):
    extract_data = df.copy()

    # List of nutritional columns (adjust based on your actual data)
    nutrition_columns = ["calories", "total fat", "sugar", "sodium", "protein", "saturated fat", "carbohydrates"]
    
    # Apply nutritional filters to the relevant columns
    for column, maximum in zip(nutrition_columns, max_nutrition):
        extract_data = extract_data[extract_data[column] < maximum]
    
    # Filter based on food type if provided
    if food_type is not None:
        extract_data = extract_data[extract_data[food_type] == True]

    # Filter based on ingredients if provided
    if ingredient_filter is not None:
        for ingredient in ingredient_filter:
            extract_data = extract_data[extract_data["ingredients"].str.contains(ingredient, regex=False)]

    return extract_data

# Apply pipeline transformation
def apply_pipeline(pipeline, _input, extract_data):
    _input = np.array(_input).reshape(1, -1)
    return extract_data.iloc[pipeline.transform(_input)[0]]

# Recommendation function
def recommend(df, max_nutritional_values, food_type, ingredient_filter=[], params={"return_distance": False}):
    # Apply filtering based on input
    extract_data = filter_data(df, ingredient_filter, max_nutritional_values, food_type)
    
    # Scale the data
    prep_data, scaler = scaling(extract_data)
    
    # Build and apply nearest neighbors model
    neigh = nn_predictor(prep_data)
    pipeline = build_pipeline(neigh, scaler, params)
    
    return apply_pipeline(pipeline, max_nutritional_values, extract_data)

# FastAPI app
app = FastAPI()

# Request and response models
class RecommendationRequest(BaseModel):
    max_nutritional_values: List[float]
    food_type: Optional[str] = None
    ingredient_filter: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    recommended_foods: List[Dict[str, str]]

# Endpoint to get all data
@app.get("/")
async def get_all_data():
    try:
        # Display only a few lines (e.g., first 5 rows) from the filtered data
        sample_data = filter_data(df, ingredient_filter=[], max_nutrition=[float('inf')]*7, food_type=None).head(5)
        data = sample_data.to_dict(orient="records")
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Endpoint to get recommendations based on user input
@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommend(
            df,
            request.max_nutritional_values,
            request.food_type,
            request.ingredient_filter,
        )
        
        # Convert the entire row to a dictionary and convert all values to strings
        recommended_foods = recommendations.applymap(str).to_dict(orient="records")
        
        return RecommendationResponse(recommended_foods=recommended_foods)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
