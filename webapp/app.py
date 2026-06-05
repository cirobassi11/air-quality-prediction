from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Carica il modello XGBoost addestrato
with open('model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

STATIONS = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

WD_OPTIONS = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']

STATION_ENC = { 'Aotizhongxin': 83.4, 'Changping': 71.2, 'Dingling': 63.8, 'Dongsi': 85.1, 'Guanyuan': 82.7, 'Gucheng': 88.3, 'Huairou': 66.4, 'Nongzhanguan': 84.9, 'Shunyi': 74.1, 'Tiantan': 83.6, 'Wanliu': 82.1, 'Wanshouxigong': 84.2 }

# Direzione del vento in gradi
WD_TO_DEG = { 'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5 }

@app.route('/') # Quando l'utente accede a http://localhost:5000/ esegue la funzione index()
def index():
    # Restituisce la pagina index.html passando le stazioni e le opzioni di direzione del vento
    return render_template('index.html', stations=STATIONS, wd_options=WD_OPTIONS)

@app.route('/predict', methods=['POST']) # Quando l'utente invia una richiesta POST a http://localhost:5000/predict esegue la funzione predict()
def predict():
    try:
        d = request.get_json() # Dati in formato JSON inviati dal client

        hour  = int(d['hour'])
        month = int(d['month'])
        dow   = int(d['dow'])

        # Controllo della correttezza dell'input
        if not (0 <= hour <= 23):
            return jsonify({'error': 'Ora non valida (0–23)'}), 400
        if not (1 <= month <= 12):
            return jsonify({'error': 'Mese non valido (1–12)'}), 400
        if not (0 <= dow <= 6):
            return jsonify({'error': 'Giorno settimana non valido (0–6)'}), 400
        for field in ['SO2', 'NO2', 'CO', 'O3', 'RAIN', 'WSPM']:
            if float(d[field]) < 0:
                return jsonify({'error': f'{field} non può essere negativo'}), 400

        # Codifica le variabili cicliche
        cyc = lambda v, p: (np.sin(2 * np.pi * v / p), np.cos(2 * np.pi * v / p))

        # Codifica ciclica di ora, mese e giorno settimana
        hour_s, hour_c   = cyc(hour, 24)
        month_s, month_c = cyc(month, 12)
        dow_s, dow_c     = cyc(dow, 7)

        # Codifica ciclica della direzione del vento
        wd_rad = np.deg2rad(WD_TO_DEG.get(d['wd'], 0))
        wd_s, wd_c = np.sin(wd_rad), np.cos(wd_rad)

        # Vettore di input per il modello
        features = np.array([[
            float(d['SO2']),
            float(d['NO2']),
            float(d['CO']),
            float(d['O3']),
            float(d['TEMP']),
            float(d['PRES']),
            float(d['DEWP']),
            float(d['RAIN']),
            float(d['WSPM']),
            1 if dow >= 5 else 0,
            hour_s, hour_c,
            month_s, month_c,
            dow_s, dow_c,
            wd_s, wd_c,
            STATION_ENC.get(d['station'], 80.0)
        ]])

        # Previsione del PM2.5 con il modello caricato
        return jsonify({'pm25': max(0.0, float(model.predict(features)[0]))})

    except Exception as ex:
        return jsonify({'error': str(ex)}), 400

if __name__ == '__main__':
    app.run(debug=False)