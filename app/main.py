from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json
import numpy as np
import shutil


app = FastAPI()

model_path_src = 'model/model.h5'
recipe_path_src = 'model/recipe.json'
tags_path_src = 'model/tag.json'
word_index_path_src = 'model/word_index.json'

model_path = '/tmp/model.h5'
recipe_path = '/tmp/recipe.json'
tags_path = '/tmp/tag.json'
word_index_path = '/tmp/word_index.json'

shutil.copy2(model_path_src, model_path)
shutil.copy2(recipe_path_src, recipe_path)
shutil.copy2(tags_path_src, tags_path)
shutil.copy2(word_index_path_src, word_index_path)

MAX_INGREDIENTS = 2
MAX_WORDS_IN_INGREDIENT = 10
MAX_TAGS = 200
MAX_VOCAB_SIZE = 10000

with open(recipe_path, 'r', encoding='utf8') as f:
    data = json.load(f)


class MessagesReq(BaseModel):
    messages: str


def load_tags():
    try:
        f = open(tags_path, 'r')
        tags = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        tags = []
    return tags


def save_tags(tags):
    tags_file = open(tags_path, 'w+')
    tags_file.write(json.dumps(tags))
    tags_file.close()


def load_word_index():
    try:
        f = open(word_index_path, 'r')
        word_index = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        word_index = {'<PAD>': 0, '<OOV>': 1}
    return word_index


def save_word_index(word_index):
    word_index_file = open(word_index_path, 'w+')
    word_index_file.write(json.dumps(word_index))
    word_index_file.close()


tags = load_tags()
model = keras.models.load_model(model_path)

# Create a dictionary based on its tag
recipe_ids = {obj["tag"]: obj["id"] for obj in data}
recipe_name = {obj["tag"]: obj["name"] for obj in data}
ingredients = {obj["tag"]: obj["ingredients_raw_str"] for obj in data}
serving_size = {obj["tag"]: obj["serving_size"] for obj in data}
servings = {obj["tag"]: obj["servings"] for obj in data}
instruction = {obj["tag"]: obj["steps"] for obj in data}
calories = {obj["tag"]: obj["calories"] for obj in data}
image = {obj["tag"]: obj["food_image"] for obj in data}


def preprocess_input(user_input):
    tokenizer = keras.preprocessing.text.Tokenizer(lower=True, filters=' ')

    word_index = load_word_index()
    for recipe in data:
        for ingredient in recipe['ingredients']:
            tokenizer.fit_on_texts([ingredient])

    word_index.update(tokenizer.word_index)
    save_word_index(word_index)

    # Tokenizing and padding the user input
    user_data = tokenizer.texts_to_sequences(user_input)
    user_data = [item for sublist in user_data for item in sublist]  # Flatten the nested list

    # Padding the number of ingredients and words in the user data
    user_data = pad_sequences([user_data], padding='post', maxlen=MAX_INGREDIENTS * MAX_WORDS_IN_INGREDIENT, value=0)

    return user_data


def get_top_labels(model, preprocessed_input, tags, top_k=5):
    predictions = model.predict(preprocessed_input)
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    top_labels = [tags[index] for index in top_indices]
    return top_labels, top_indices


def predict_labels(ingredients_list):
    preprocessed_input = preprocess_input(ingredients_list)
    top_labels, top_indices = get_top_labels(model, preprocessed_input, tags, top_k=5)
    return top_labels, top_indices


@app.get("/")
async def root():
    return {"message": "Bangkit 2023"}


@app.post("/predict/")
async def predict_label(req: MessagesReq):
    top_labels, top_indices = predict_labels(req.messages.split(','))

    predictions = []
    for label in top_labels:
        prediction = {
            "id": recipe_ids[label],
            "recipe_name": recipe_name[label],
            "ingredients": ingredients[label],
            "serving_size": serving_size[label],
            "servings": servings[label],
            "instruction": instruction[label],
            "calories": calories[label],
            "image": image[label]
        }
        predictions.append(prediction)

    return {"predictions": predictions}

recipe_ids = {obj["tag"]: obj["id"] for obj in data}

@app.post("/recipe/{id}")
async def get_recipe_by_id(id: int):
    if id in recipe_ids.values():
        label = next(key for key, value in recipe_ids.items() if value == id)
        recipe = {
            "id": id,
            "recipe_name": recipe_name[label],
            "ingredients": ingredients[label],
            "serving_size": serving_size[label],
            "servings": servings[label],
            "instruction": instruction[label],
            "calories": calories[label],
            "image": image[label]
        }
        return recipe
    else:
        return {"message": "Recipe not found"}