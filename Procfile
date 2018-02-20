web: gunicorn -t 120 --pythonpath dogapp dogapp:app --log-file=-
heroku ps:scale web=1
