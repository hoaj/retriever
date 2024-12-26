# Use the official Python image as a parent image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Install PostgreSQL client libraries
RUN apt-get update && apt-get install -y libpq-dev

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port that the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.fastapi.serve:app", "--host", "0.0.0.0", "--port", "8000"]
