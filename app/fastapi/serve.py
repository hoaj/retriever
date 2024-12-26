from fastapi import FastAPI
from app.fastapi.routes import setup_routes
from fastapi.responses import RedirectResponse
import uvicorn

app = FastAPI()

routes = setup_routes()
app.include_router(routes)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
