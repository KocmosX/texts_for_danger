from collections import Counter
import pandas as pd
import numpy as np
import torch
from sklearn.utils import shuffle
from Neural_Architecture import LSTM_architecture
import string
import re
import json

# Чтение и предобработка данных для токенизации
def read_data():
    n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
    data_positive = pd.read_csv('./dataset/positive.csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'], encoding='cp1251')
    data_negative = pd.read_csv('./dataset/negative.csv', sep=';', on_bad_lines='skip', names=n, usecols=['text'])

    sample_size = 50000
    texts_withoutshuffle = np.concatenate((data_positive['text'].values[:sample_size],
                                           data_negative['text'].values[:sample_size]), axis=0)
    labels_withoutshuffle = np.asarray([1] * sample_size + [0] * sample_size)
    assert len(texts_withoutshuffle) == len(labels_withoutshuffle)
    texts, labels = shuffle(texts_withoutshuffle, labels_withoutshuffle, random_state=0)

    return texts, labels

texts, labels = read_data()

# Функция для токенизации текста
def tokenize():
    punctuation = string.punctuation
    all_texts = ' separator '.join(texts)
    all_texts = all_texts.lower()
    all_text = ''.join([c for c in all_texts if c not in punctuation])
    texts_split = all_text.split(' separator ')
    all_text = ' '.join(texts_split)
    words = all_text.split()
    return words

# Получение словаря
def get_vocabulary():
    words = tokenize()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    return vocab, vocab_to_int

# Загрузка словаря
with open('vocab_to_int.json', 'r') as f:
    vocab_to_int = json.load(f)

# Токенизация текста для предсказания
def tokenize_text(test_text):
    if not isinstance(test_text, str):
        test_text = str(test_text)
    punctuation = string.punctuation
    test_text = test_text.lower()
    test_text = ''.join([c for c in test_text if c not in punctuation])
    test_words = test_text.split()
    new_text = [word for word in test_words if (word[0] != '@') and ('http' not in word) and (not word.isdigit())]

    test_ints = []
    mas_to_int = [vocab_to_int[word] for word in new_text if word in vocab_to_int]
    test_ints.append(mas_to_int)

    return test_ints

# Функция добавления паддингов
def add_pads(texts_ints, seq_length):
    features = np.zeros((len(texts_ints), seq_length), dtype=int)
    for i, row in enumerate(texts_ints):
        if len(row) == 0:
            continue
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

# Загрузка и предсказание с помощью модели
def predict(model, test_text, sequence_length=50):  # Увеличиваем длину обрабатываемого текста
    model.eval()
    test_ints = tokenize_text(test_text)
    seq_length = sequence_length
    features = add_pads(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)
    h = model.init_hidden_state(batch_size)
    output, h = model(feature_tensor, h)

    pred = torch.round(output.squeeze())
    pos_prob = output.item()

    if pred.item() == 1:
        result = "Опасное сообщение"
    else:
        result = "Безопасное сообщение"

    return result, pos_prob

# Замените текущие значения на следующие
vocab_size = len(vocab_to_int) + 1  # Это значение должно совпадать с используемым при обучении
embedding_dim = 100  # Это значение должно совпадать с используемым при обучении
output_size = 1
hidden_dim = 128
number_of_layers = 3  # Увеличиваем количество слоев LSTM

# Создание и загрузка модели
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=True))

# Пример предсказания
test_text = "ТАСС: освобождение Шевченко в ДНР открыло путь на Красноармейск Освобождение поселка Шевченко в ДНР позволило открыть российским войскам дорогу на Красноармейск (украинское название - Покровск) сообщили ТАСС в силовых структурах.ТАСС Сообщается о поражении трех механизированных бригад ВСУ и бригады нацгвардии под Срибным, Алексеевкой, Гришино и Белицким в ДНР.Газета.Ru Украинская армия потеряла на этом участке фронта до 535 военнослужащих, танк Leopard производства ФРГ, четыре боевые бронированные машины, три автомобиля и пять артиллерийских орудий.Интерфакс Подразделения группировки войск Днепр нанесли поражение живой силе и технике двух механизированных и пехотной бригад ВСУ в районах населенных пунктов Блакитное, Приморское и Новояковлевка Запорожской области.Интерфакс. Фейк."
type_of_tonal, pos_prob = predict(model, test_text)
print(f"Опасность текста - {type_of_tonal}, Риск = {round(pos_prob * 100)}%")
