from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from .utils import postprocessing, preprocessing
import numpy as np
import os
import os.path
from tensorflow import keras

IMAGE_DIM = 299
nsfw_model = settings.NSFW_MODEL
similarity_model = settings.SIMILARITY_MODEL
keyword_extractor_model = settings.KEYWORD_EXTRACTOR_MODEL

@api_view(['GET'])
def home(request):
    api_overview = {
        '/nsfw' : 'Expects a image'
    }
    return Response(api_overview)

def preprocess_image(img, image_size, verbose=True):
    image = keras.preprocessing.image.load_img(img, target_size=image_size)
    image = keras.preprocessing.image.img_to_array(image)
    image /= 255
    return np.asarray([image])

def classify_image(image):
    categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    model_preds = nsfw_model.predict(image)
    single_probs = {}
    for i, single_preds in enumerate(model_preds[0]):
        single_probs[categories[i]] = float(single_preds)
    return single_probs


@api_view(['POST'])
def nsfw(req):
    image = req.FILES['image']
    path = default_storage.save('image.jpg', ContentFile(image.read()))
    img_file = os.path.join(settings.MEDIA_ROOT, path)
    processed_image = preprocess_image(img_file, (IMAGE_DIM, IMAGE_DIM))
    image_pred = classify_image(processed_image)
    api_overview = {
        'success' : '1',
        'result': image_pred
    }
    return Response(api_overview)


@api_view(['POST'])
def similarity(req):
    MAX_NB_WORDS = 200000
    MAX_SEQUENCE_LENGTH = 255
    questions = [req.data['q1'], req.data['q2']]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)

    question1_word_sequences = tokenizer.texts_to_sequences([questions[0]])
    question2_word_sequences = tokenizer.texts_to_sequences([questions[1]])
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    out = similarity_model.predict([np.array(q1_data), np.array(q2_data)], steps=None, callbacks = None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    
    api_overview = {
        'success' : '1',
        'result': out[0][0]
    }
    
    return Response(api_overview)

@api_view(['POST'])
def keyword(req):
    MAX_VOCABULARY_SIZE = 200000
    MAX_DOCUMENT_LENGTH = 550
    text = {'text': req.data['text']}
    test_doc = {'text': text_to_word_sequence(text['text'])}

    tokenizer = Tokenizer(num_words=MAX_VOCABULARY_SIZE)
    tokenizer.fit_on_texts([text['text']])

    test_x = tokenizer.texts_to_sequences([text['text']])
    test_x = pad_sequences(test_x, maxlen=MAX_DOCUMENT_LENGTH)
    output = keyword_extractor_model.predict([np.array(test_x)], steps=None, callbacks = None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    obtained_tokens = postprocessing.undo_sequential(output)
    obtained_words = postprocessing.get_words(test_doc, obtained_tokens)
    clean_words = postprocessing.get_valid_patterns(obtained_words)

    clean_words = clean_words['text']

    keyphrase_2, keyphrase_more = [], []
    for clean_word in clean_words:
        if len(clean_word) <= 2:
            keyphrase_2.append(' '.join(clean_word))

        keyphrase_more.append(' '.join(clean_word))
    api_overview = {
        'success' : '1',
        'less than 3': keyphrase_2,
        'all' : keyphrase_more
    }
    return Response(api_overview)