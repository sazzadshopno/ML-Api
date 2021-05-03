from pathlib import Path
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
import nltk

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-e8^sx7j&voq6_yc0_e2@t05#8#=gdqzm6epz2qb447(7f8pnp3'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/

STATIC_URL = '/static/'

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny'
    ]
}

MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

NSFW_MODEL = load_model('nsfw.299x299.h5')

SIMILARITY_MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'
SIMILARITY_MODEL_JSON_FILE = 'question_pairs_json.json'
KEYWORD_EXTRACTOR_MODEL_WEIGHTS_FILE = "keyphrase_weights.h5"
KEYWORD_EXTRACTOR_MODEL_JSON_FILE = 'keyphrase_weights.json'

def init(): 
    json_file = open(SIMILARITY_MODEL_JSON_FILE, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    similarity_model = model_from_json(loaded_model_json)
    similarity_model.load_weights(SIMILARITY_MODEL_WEIGHTS_FILE)
    similarity_model.compile(loss='binary_crossentropy',optimizer=SGD(lr=.01, clipvalue=0.5),metrics=['accuracy'])
    json_file = open(KEYWORD_EXTRACTOR_MODEL_JSON_FILE,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    keyword_extractor_model = model_from_json(loaded_model_json)
    keyword_extractor_model.load_weights(KEYWORD_EXTRACTOR_MODEL_WEIGHTS_FILE)
    keyword_extractor_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'],
                  sample_weight_mode="temporal")
    return similarity_model, keyword_extractor_model

similarity_model, keyword_extractor_model = init()

SIMILARITY_MODEL = similarity_model
KEYWORD_EXTRACTOR_MODEL = keyword_extractor_model