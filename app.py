from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('admission_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------- Page Routes ----------

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Here you can add logic to store user info if needed
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/login', methods=['POST'])
def login():
    # Placeholder: You can add user validation here
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

# ---------- Prediction Route ----------

@app.route('/submit', methods=['POST'])
def submit():
    try:
        gender = int(request.form['gender'])
        hsc_p = float(request.form['hsc_p'])
        hsc_b_Central = int(request.form['hsc_b'])
        hsc_s = request.form['hsc_s']

        # One-hot encoding for stream
        if hsc_s == "Commerce":
            commerce = 1
            science = 0
        elif hsc_s == "Science":
            commerce = 0
            science = 1
        else:
            commerce = 0
            science = 0

        kcet_score = float(request.form['kcet_score'])
        university_rating = int(request.form['university_rating'])

        # Prepare input
        input_data = np.array([gender, hsc_p, hsc_b_Central, commerce, science, kcet_score, university_rating]).reshape(1, -1)
        scaled = scaler.transform(input_data)
        prediction = round(model.predict(scaled)[0], 2)

        return render_template('prediction.html', result=prediction)

    except Exception as e:
        return f"Error occurred: {e}", 400

# ---------- Run the App ----------

if __name__ == '__main__':
    app.run(debug=True)
