FROM python:3.8
WORKDIR /env
COPY requirements.txt /env/
RUN pip install -r requirements.txt
COPY . /env
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
