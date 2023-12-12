FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt



COPY ./bank_logo.jpg /code/bank_logo.jpg
COPY ./client_full_data.csv /code/client_full_data.csv
COPY ./*.py /code/

CMD ["streamlit", "run", "main.py"]