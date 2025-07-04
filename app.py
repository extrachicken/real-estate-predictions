import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Используем не-интерактивный бэкенд
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Для работы flash-сообщений

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 
                          'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Отсутствуют следующие колонки: {', '.join(missing_columns)}"
        
        df = df[required_columns]
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv'), index=False)
        return True, "Данные успешно обработаны"
    except Exception as e:
        return False, f"Ошибка при обработке файла: {str(e)}"

def generate_plots():
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv'))
    
    # График распределения цен
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Распределение цен на недвижимость')
    plt.xlabel('Цена')
    plt.ylabel('Частота')
    
    # Сохраняем график в base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    price_distribution = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Корреляционная матрица
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.title('Корреляционная матрица')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    correlation_matrix = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return price_distribution, correlation_matrix

def evaluate_prediction(prediction, features):
    try:
        # Загружаем обработанные данные
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.csv'))
        
        # Находим похожие объекты (в пределах 20% от введенных параметров)
        similar_houses = df[
            (df['bedrooms'].between(features[0] * 0.8, features[0] * 1.2)) &
            (df['bathrooms'].between(features[1] * 0.8, features[1] * 1.2)) &
            (df['sqft_living'].between(features[2] * 0.8, features[2] * 1.2)) &
            (df['sqft_lot'].between(features[3] * 0.8, features[3] * 1.2)) &
            (df['floors'] == features[4]) &
            (df['waterfront'] == features[5]) &
            (df['view'].between(features[6] - 1, features[6] + 1)) &
            (df['condition'].between(features[7] - 1, features[7] + 1))
        ]
        
        # Статистический анализ
        mean_price = df['price'].mean()
        std_price = df['price'].std()
        
        # Вычисляем z-score для предсказанной цены
        z_score = (prediction - mean_price) / std_price
        
        # Вычисляем p-value для двустороннего теста
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Определяем, является ли предсказание статистически значимым
        is_significant = p_value < 0.05
        
        # Вычисляем доверительный интервал для среднего значения цен
        confidence_interval = stats.norm.interval(0.95, loc=mean_price, scale=std_price/np.sqrt(len(df)))
        
        # Вычисляем процентиль предсказанной цены
        percentile = stats.percentileofscore(df['price'], prediction)
        
        if len(similar_houses) > 0:
            # Вычисляем метрики для похожих объектов
            mae = mean_absolute_error(similar_houses['price'], [prediction] * len(similar_houses))
            mape = mean_absolute_percentage_error(similar_houses['price'], [prediction] * len(similar_houses))
            
            # Находим ближайшие объекты
            similar_houses['price_diff'] = abs(similar_houses['price'] - prediction)
            closest_houses = similar_houses.nsmallest(3, 'price_diff')
            
            return {
                'mae': mae,
                'mape': mape * 100,  # Переводим в проценты
                'similar_houses': closest_houses[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']].to_dict('records'),
                'total_similar': len(similar_houses),
                'statistical_analysis': {
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'z_score': z_score,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'confidence_interval': confidence_interval,
                    'percentile': percentile
                }
            }
        else:
            return {
                'mae': None,
                'mape': None,
                'similar_houses': [],
                'total_similar': 0,
                'statistical_analysis': {
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'z_score': z_score,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'confidence_interval': confidence_interval,
                    'percentile': percentile
                }
            }
    except Exception as e:
        print(f"Ошибка при оценке предсказания: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Файл не выбран')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        success, message = process_dataset(file_path)
        if success:
            return redirect(url_for('prediction_form'))
        else:
            flash(message)
            return redirect(url_for('index'))
    
    flash('Недопустимый формат файла')
    return redirect(url_for('index'))

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_form():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['bedrooms']),
                float(request.form['bathrooms']),
                float(request.form['sqft_living']),
                float(request.form['sqft_lot']),
                float(request.form['floors']),
                float(request.form['waterfront']),
                float(request.form['view']),
                float(request.form['condition'])
            ]
            
            model = joblib.load('best_model.joblib')
            prediction = model.predict([features])[0]
            
            # Оцениваем точность предсказания
            evaluation = evaluate_prediction(prediction, features)
            
            price_distribution, correlation_matrix = generate_plots()
            
            return render_template('results.html', 
                                 prediction=prediction,
                                 price_distribution=price_distribution,
                                 correlation_matrix=correlation_matrix,
                                 evaluation=evaluation)
        except Exception as e:
            flash(f'Ошибка при предсказании: {str(e)}')
            return redirect(url_for('prediction_form'))
    
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
