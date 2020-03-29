# Python image to use.
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN python -m pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8080

# Install ptvsd for debugging
RUN pip install ptvsd

# Copy the rest of the working directory contents into the container at /app
COPY . /app

# Start the server when the container launches
#CMD ["python", "-m", "ptvsd", "--port", "8501", "--host", "0.0.0.0", "churn.py", "runserver", "0.0.0.0:8080", "--noreload"]
CMD ["streamlit", "run", "--server.port", "8080", "Churn.py"]
