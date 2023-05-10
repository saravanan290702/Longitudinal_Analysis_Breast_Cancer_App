from flask import Flask,request,jsonify
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():

    time = int(request.form.get('time'))
    radius_mean = float(request.form.get('radius_mean'))
    texture_mean = float(request.form.get('texture_mean'))
    smoothness_mean = float(request.form.get('smoothness_mean'))
    compactness_mean = float(request.form.get('compactness_mean'))
    concave_points_mean = float(request.form.get('concave points_mean'))
    symmetry_mean = float(request.form.get('symmetry_mean'))
    fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))
    texture_se = float(request.form.get('texture_se'))
    perimeter_se = float(request.form.get('perimeter_se'))
    smoothness_se = float(request.form.get('smoothness_se'))
    concavity_se = float(request.form.get('concavity_se'))
    concave_points_se = float(request.form.get('concave points_se'))
    symmetry_se = float(request.form.get('symmetry_se'))
    fractal_dimension_se = float(request.form.get('fractal_dimension_se'))
    smoothness_worst = float(request.form.get('smoothness_worst'))
    compactness_worst = float(request.form.get('compactness_worst'))
    concave_points_worst = float(request.form.get('concave points_worst'))
    symmetry_worst = float(request.form.get('symmetry_worst'))
    Tumor_Size = float(request.form.get('Tumor Size'))
    Lymph_node_status = float(request.form.get('Lymph node status'))
    input_query = np.array([time, radius_mean, texture_mean, smoothness_mean, compactness_mean, concave_points_mean, symmetry_mean,
         fractal_dimension_mean, texture_se, perimeter_se, smoothness_se, concavity_se, concave_points_se, symmetry_se,
         fractal_dimension_se, smoothness_worst, compactness_worst, concave_points_worst, symmetry_worst, Tumor_Size,
         Lymph_node_status]).reshape(1, 21)
    result = model.predict(input_query)[0]
    if result[0] == "N":
        result = "Breast Cancer Will Not Re-Occur for this conditions"
    else:
        result = "Breast Cancer Will  Re-Occur for this conditions"
    return jsonify({'Result': str(result)})

if __name__ == '__main__':
    app.run(debug=True)