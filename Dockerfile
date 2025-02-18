# python runtime as a parent image
FROM python:3.9-slim

# set the working director in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
#RUN pip install -r requirements.txt
# while rebuilding
RUN pip install -r requirements.txt && rm -rf /root/.cache


# Copy the entire project into the container
COPY . .
COPY /weights/model_epoch_1.pt /app/model_epoch_1.pt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


