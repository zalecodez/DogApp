web: gunicorn -t 120 --pythonpath dogapp dogapp:app --log-file=- --preload
heroku ps:scale web=1
