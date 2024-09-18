import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def train_models(train_data):

    X = train_data[['depth']].values
    y_interdepth = train_data['pred_Interdepth'].values
    y_intradepth = train_data['pred_Intradepth'].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    model_interdepth = SVR(kernel='rbf')
    model_interdepth.fit(X_scaled, y_interdepth)

    model_intradepth = SVR(kernel='rbf')
    model_intradepth.fit(X_scaled, y_intradepth)

    joblib.dump(scaler_X, 'scaler_X.pkl')
    return model_interdepth, model_intradepth

def predict(models, test_data):

    X_test = test_data[['depth']].values
    scaler_X = joblib.load('scaler_X.pkl')
    X_test_scaled = scaler_X.transform(X_test)

    predictions = {}
    for key, model in zip(['pred_Interdepth', 'pred_Intradepth'], models):
        predictions[key] = model.predict(X_test_scaled)
    return predictions

def save_predictions_to_csv(test_data, predictions, output_file):
    for key in predictions:
        test_data[key] = predictions[key]

    test_data.to_csv(output_file, index=False)

def main():

    train_data = load_data('depth_train_unique.csv')

    model_interdepth, model_intradepth = train_models(train_data)

    joblib.dump(model_interdepth, 'model_interdepth.pkl')
    joblib.dump(model_intradepth, 'model_intradepth.pkl')

    test_data = load_data('depth_test.csv')

    predictions = predict([model_interdepth, model_intradepth], test_data)

    output_file = 'svr_depth_summary_val_with_predictions.csv'
    save_predictions_to_csv(test_data, predictions, output_file)

    print(f"预测结果已保存至 {output_file}")

if __name__ == '__main__':
    main()