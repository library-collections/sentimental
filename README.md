sentimental
===========

basic online sentiment analysis demo

## Installation

```
pip install -r requirements.txt
```

Add a `config.py` with the following:

```
CSRF_ENABLED = True
SECRET_KEY = 'you-will-never-guess'
SAVE_PATH = '/path/to/where/you/want/to/back/up/the/classifier'
```

And that should be it!

```
python sentimental.py
```
