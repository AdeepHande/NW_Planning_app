FROM python:3.8.5

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install streamlit
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["/app/src/multipage.py"]

RUN bash -c 'echo -e"\
[server]\n\
enableCORS = false\n\
" > /app/streamlit/config.toml'

