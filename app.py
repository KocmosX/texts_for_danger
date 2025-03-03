from flask import Flask, render_template, request, redirect, url_for, session
from flask_socketio import SocketIO, emit
import torch
from test_model import predict
from Neural_Architecture import LSTM_architecture
from math import ceil
import re
import json
import pandas as pd
from werkzeug.utils import secure_filename
import os
import tempfile

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'xls'}

socketio = SocketIO(app)

# Загрузка словаря
with open('vocab_to_int.json', 'r') as f:
    vocab_to_int = json.load(f)

vocab_size = len(vocab_to_int) + 1  # Используем то же значение, что и в обученной модели
output_size = 1
embedding_dim = 100  # Используем то же значение, что и в обученной модели
hidden_dim = 128
number_of_layers = 3  # Увеличиваем количество слоев LSTM
model = LSTM_architecture(vocab_size, output_size, embedding_dim, hidden_dim, number_of_layers)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu'), weights_only=True))
seq_length = 50  # Увеличиваем длину обрабатываемого текста

def validate_text(text):
    # Убираем проверку на наличие только русских символов и цифр
    return True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=['GET', 'POST'])
def hello():
    flag = False
    type_of_tonal = ""
    prob = 0
    name = ""
    error_message = ""
    dangerous_words_found = []
    if request.method == 'POST':
        flag = True
        if request.form["submit_button"]:
            name1 = request.form['text_tonal']
            if len(name1) != 0:
                name = name1
                if not validate_text(name):
                    error_message = "Пожалуйста, введите корректный текст."
                else:
                    type_of_tonal, pos_prob = predict(model, name, seq_length)
                    dangerous_words_found = model.detect_dangerous_words(name)
                    if dangerous_words_found:
                        type_of_tonal = "Потенциально опасное сообщение"
                        prob = 100  # Если есть опасные слова, вероятность 100%
                    else:
                        type_of_tonal = "Опасность сообщения"
                        prob = 0  # Если нет опасных слов, вероятность 0%
            else:
                error_message = "Пожалуйста, введите текст."

    return render_template('main.html', flag=flag, type_of_tonal=type_of_tonal, percent="{} %".format(prob), text=name, error_message=error_message, dangerous_words=dangerous_words_found)

@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df = pd.read_excel(filepath)
            total_rows = len(df)
            results = []

            for index, row in df.iterrows():
                text = str(row.iloc[0])  # Преобразуем значение в строку
                if validate_text(text):
                    type_of_tonal, pos_prob = predict(model, text, seq_length)
                    dangerous_words_found = model.detect_dangerous_words(text)
                    if dangerous_words_found:
                        type_of_tonal = "Потенциально опасное сообщение"
                        prob = 100  # Если есть опасные слова, вероятность 100%
                    else:
                        type_of_tonal = "Безопасное сообщение"
                        prob = 0  # Если нет опасных слов, вероятность 0%
                    results.append({
                        'text': text,
                        'type_of_tonal': type_of_tonal,
                        'prob': prob,
                        'dangerous_words': dangerous_words_found
                    })
                else:
                    results.append({
                        'text': text,
                        'error_message': "Пожалуйста, введите корректный текст."
                    })

                # Отправка прогресса через WebSocket
                progress = (index + 1) / total_rows * 100
                socketio.emit('progress', {'progress': progress})

            # Сортировка результатов по уровню опасности
            results.sort(key=lambda x: x['prob'], reverse=True)

            # Сохранение результатов во временный файл
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            with open(temp_file.name, 'w') as f:
                json.dump(results, f)
            session['results_file'] = temp_file.name

            return redirect(url_for('results'))
    return render_template('upload.html')

@app.route("/results", methods=['GET', 'POST'])
def results():
    filter_type = request.args.get('filter', 'all')
    results_file = session.get('results_file', None)
    results = []

    if results_file:
        with open(results_file, 'r') as f:
            results = json.load(f)

    total_records = len(results)
    dangerous_count = sum(1 for result in results if result['prob'] == 100)
    safe_count = total_records - dangerous_count

    if filter_type == 'dangerous':
        results = [result for result in results if result['prob'] == 100]
    elif filter_type == 'safe':
        results = [result for result in results if result['prob'] == 0]

    return render_template('results.html', results=results, filter=filter_type, total_records=total_records, dangerous_count=dangerous_count, safe_count=safe_count)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    socketio.run(app, debug=True)
