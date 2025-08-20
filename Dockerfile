FROM apache/airflow:2.10.2-python3.10

# Copy requirements
COPY requirements.txt /requirements.txt

# Install requirements
RUN pip install --no-cache-dir -r /requirements.txt
