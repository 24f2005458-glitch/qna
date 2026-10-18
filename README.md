# QnA Server Test
##Start using

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000

##Environment variable to be set
QUESTION_DATA_URL='YOUR_COMPLETE_QUESTION_DATA_URL'

