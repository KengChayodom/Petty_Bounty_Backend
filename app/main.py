from fastapi import FastAPI

from app.routers import missing_pets,upload,sightings

app = FastAPI(title="Petty Bounty API")

# app.include_router(users.router)
app.include_router(missing_pets.router) 
app.include_router(upload.router)
app.include_router(sightings.router)

@app.get("/")
def root():
    return {"message": "Petty Bounty API is running! on http://127.0.0.1:8000/docs"}