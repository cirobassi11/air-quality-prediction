from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model_xgb.pkl')

STATIONS = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']

WD_OPTIONS = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW', 'SW', 'W', 'WNW', 'WSW']

STATION_ENC = {
    'Aotizhongxin': 83.4, 'Changping': 71.2, 'Dingling': 63.8,
    'Dongsi': 85.1, 'Guanyuan': 82.7, 'Gucheng': 88.3,
    'Huairou': 66.4, 'Nongzhanguan': 84.9, 'Shunyi': 74.1,
    'Tiantan': 83.6, 'Wanliu': 82.1, 'Wanshouxigong': 84.2
}

WD_ENC = {wd: i for i, wd in enumerate(WD_OPTIONS)}

AQI_BINS = [
    (50,  'Eccellente',              '#2ecc71'),
    (100, 'Buono',                   '#f1c40f'),
    (150, 'Leggermente Inquinato',   '#e67e22'),
    (200, 'Moderatamente Inquinato', '#e74c3c'),
    (300, 'Fortemente Inquinato',    '#9b59b6'),
    (float('inf'), 'Seriamente Inquinato', '#922b21'),
]

HTML = """
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>PM2.5 Predictor</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 1rem; }
    h1   { font-size: 1.6rem; margin-bottom: 0.3rem; }
    p    { color: #666; margin-bottom: 2rem; font-size: 0.9rem; }
    h3   { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: #999; margin: 1.5rem 0 0.8rem; border-bottom: 1px solid #eee; padding-bottom: 0.4rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.8rem; }
    label { display: flex; flex-direction: column; font-size: 0.8rem; color: #555; gap: 0.3rem; }
    input, select { padding: 0.45rem 0.6rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9rem; width: 100%; }
    input:focus, select:focus { outline: none; border-color: #333; }
    button { margin-top: 1.5rem; padding: 0.7rem 2rem; background: #222; color: #fff; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #444; }
    #result { margin-top: 2rem; padding: 1.2rem 1.5rem; border-radius: 6px; display: none; background: #f5f5f5; }
    #result-value { font-size: 3rem; font-weight: 700; line-height: 1; }
    #result-unit  { font-size: 0.8rem; color: #666; margin-top: 0.2rem; }
    #aqi-badge    { display: inline-block; margin-top: 0.8rem; padding: 0.25rem 0.8rem; border-radius: 100px; font-size: 0.8rem; font-weight: 600; }
    #error { margin-top: 1rem; color: #e74c3c; font-size: 0.85rem; display: none; }
  </style>
</head>
<body>
  <h1>PM2.5 Predictor</h1>
  <p>Inserisci i parametri per stimare la concentrazione di PM2.5 (µg/m³).</p>

  <form id="form">
    <h3>Stazione e Tempo</h3>
    <div class="grid">
      <label>Stazione
        <select name="station">
          {% for s in stations %}<option value="{{ s }}">{{ s }}</option>{% endfor %}
        </select>
      </label>
      <label>Ora (0–23)<input type="number" name="hour" min="0" max="23" value="12"></label>
      <label>Mese (1–12)<input type="number" name="month" min="1" max="12" value="6"></label>
      <label>Giorno settimana (0=Lun)<input type="number" name="dow" min="0" max="6" value="0"></label>
    </div>

    <h3>Inquinanti</h3>
    <div class="grid">
      <label>SO₂ (µg/m³)<input type="number" name="SO2" step="0.1" value="15"></label>
      <label>NO₂ (µg/m³)<input type="number" name="NO2" step="0.1" value="50"></label>
      <label>CO (µg/m³)<input type="number" name="CO" step="1" value="800"></label>
      <label>O₃ (µg/m³)<input type="number" name="O3" step="0.1" value="60"></label>
    </div>

    <h3>Meteorologia</h3>
    <div class="grid">
      <label>TEMP (°C)<input type="number" name="TEMP" step="0.1" value="15"></label>
      <label>PRES (hPa)<input type="number" name="PRES" step="0.1" value="1010"></label>
      <label>DEWP (°C)<input type="number" name="DEWP" step="0.1" value="5"></label>
      <label>RAIN (mm)<input type="number" name="RAIN" step="0.1" value="0"></label>
      <label>WSPM (m/s)<input type="number" name="WSPM" step="0.1" value="2"></label>
      <label>Direzione vento
        <select name="wd">
          {% for w in wd_options %}<option value="{{ w }}">{{ w }}</option>{% endfor %}
        </select>
      </label>
    </div>

    <button type="submit">Calcola PM2.5</button>
  </form>

  <div id="error"></div>

  <div id="result">
    <div style="font-size:0.75rem;color:#999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem">PM2.5 stimato</div>
    <div id="result-value">—</div>
    <div id="result-unit">µg/m³</div>
    <div id="aqi-badge"></div>
  </div>

  <script>
    const AQI = [
      { max: 50,       label: 'Eccellente',              bg: '#2ecc71', color: '#fff' },
      { max: 100,      label: 'Buono',                   bg: '#f1c40f', color: '#000' },
      { max: 150,      label: 'Leggermente Inquinato',   bg: '#e67e22', color: '#fff' },
      { max: 200,      label: 'Moderatamente Inquinato', bg: '#e74c3c', color: '#fff' },
      { max: 300,      label: 'Fortemente Inquinato',    bg: '#9b59b6', color: '#fff' },
      { max: Infinity, label: 'Seriamente Inquinato',    bg: '#922b21', color: '#fff' },
    ];

    document.getElementById('form').addEventListener('submit', async e => {
      e.preventDefault();
      const errEl = document.getElementById('error');
      errEl.style.display = 'none';
      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(Object.fromEntries(new FormData(e.target)))
        });
        const json = await res.json();
        if (json.error) { errEl.textContent = json.error; errEl.style.display = 'block'; return; }

        const val = json.pm25;
        const cat = AQI.find(a => val <= a.max);

        document.getElementById('result-value').textContent = val.toFixed(1);
        document.getElementById('result-value').style.color = cat.bg;
        const badge = document.getElementById('aqi-badge');
        badge.textContent = cat.label;
        badge.style.background = cat.bg;
        badge.style.color = cat.color;
        document.getElementById('result').style.display = 'block';
      } catch {
        errEl.textContent = 'Errore di connessione.';
        errEl.style.display = 'block';
      }
    });
  </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML, stations=STATIONS, wd_options=WD_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        hour  = int(data['hour'])
        month = int(data['month'])
        dow   = int(data['dow'])
        wd_val = WD_ENC.get(data['wd'], 0)

        features = np.array([[
            float(data['SO2']),
            float(data['NO2']),
            float(data['CO']),
            float(data['O3']),
            float(data['TEMP']),
            float(data['PRES']),
            float(data['DEWP']),
            float(data['RAIN']),
            float(data['WSPM']),
            np.sin(2 * np.pi * hour  / 24), np.cos(2 * np.pi * hour  / 24),
            np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
            1 if dow >= 5 else 0,
            np.sin(2 * np.pi * dow   / 7),  np.cos(2 * np.pi * dow   / 7),
            np.sin(2 * np.pi * wd_val / len(WD_OPTIONS)),
            np.cos(2 * np.pi * wd_val / len(WD_OPTIONS)),
            STATION_ENC.get(data['station'], 80.0)
        ]])

        return jsonify({'pm25': max(0.0, float(model.predict(features)[0]))})

    except Exception as ex:
        return jsonify({'error': str(ex)}), 400


if __name__ == '__main__':
    app.run(debug=True)