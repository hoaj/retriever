from fastapi import FastAPI
from app.routes import retrieve_routes
from fastapi.responses import RedirectResponse

app = FastAPI()

app.include_router(retrieve_routes.router)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
