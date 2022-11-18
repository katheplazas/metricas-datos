import pandas as pd
import py_eureka_client.eureka_client as eureka_client
from flask import Flask
from flask_pymongo import PyMongo
from sklearn import metrics

pd.options.display.max_columns = None
rest_port = 8053

eureka_client.init(eureka_server="http://eureka:8761/eureka",
                   app_name="metricas-datos",
                   instance_port=rest_port)

app = Flask(__name__)
app.config["MONGO_URI"] = 'mongodb://root:123456@mongo:27018/preprocesamiento?authSource=admin'  ## Remoto
# app.config["MONGO_URI"] = 'mongodb://root:123456@mongo:27017/preprocesamiento?authSource=admin'  ## Local
mongo = PyMongo(app)


@app.route('/prueba', methods=["GET"])
def prueba():
    return "connected"


@app.route('/download/database', methods=["GET"])
def get_all_data():
    data = list(mongo.db.data.find({}))
    df = pd.DataFrame(data)
    df = df.drop(['_id'], axis=1)
    to_dict = df.to_dict()
    return to_dict


@app.route('/model/metrics', methods=["GET"])
def get_metric():
    data = list(mongo.db.data.find({}))
    df = pd.DataFrame(data)
    df = df.drop(['_id'], axis=1)
    tag = df.tag.tolist()
    pred_dt = df.prediction_dt.tolist()
    pred_lr = df.prediction_lr.tolist()
    pred_rf = df.prediction_rf.tolist()
    pred_svm_linear = df.prediction_svm_linear.tolist()
    list_metrics = [pred_dt, pred_lr, pred_rf, pred_svm_linear]
    metrics = {'accuracy': [], 'precision_1': [], 'precision_0': [], 'precision_avg': [], 'recall_1': [],
               'recall_0': [], 'recall_avg': [],
               'f1_score_1': [], 'f1_score_0': [], 'f1_score_avg': [], 'confusion_matrix': [],
               'model_metric': ['DT', 'LR', 'RF', 'SVM LINEAR']}
    for i in list_metrics:
        accuracy, precision_1, precision_0, precision_avg, recall_1, recall_0, recall_avg, f1_score_1, f1_score_0, \
        f1_score_avg, \
        confusion_matrix = metric_calculate(tag, i)
        metrics['accuracy'].append(accuracy)
        metrics['precision_1'].append(precision_1)
        metrics['precision_0'].append(precision_0)
        metrics['precision_avg'].append(precision_avg)
        metrics['recall_1'].append(recall_1)
        metrics['recall_0'].append(recall_0)
        metrics['recall_avg'].append(recall_avg)
        metrics['f1_score_1'].append(f1_score_1)
        metrics['f1_score_0'].append(f1_score_0)
        metrics['f1_score_avg'].append(f1_score_avg)
        metrics['confusion_matrix'].append({'tn': int(confusion_matrix[0][0]), 'fp': int(confusion_matrix[0][1]),
                                            'fn': int(confusion_matrix[1][0]), 'tp': int(confusion_matrix[1][1])})
    return metrics


def metric_calculate(y_true, y_pred):
    print("Tag: ", y_true)
    print("Predict: ", y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision_1 = metrics.precision_score(y_true, y_pred)
    precision_0 = metrics.precision_score(y_true, y_pred, pos_label=0)
    precision_avg = metrics.precision_score(y_true, y_pred, average='macro')
    recall_1 = metrics.recall_score(y_true, y_pred)
    recall_0 = metrics.recall_score(y_true, y_pred, pos_label=0)
    recall_avg = metrics.recall_score(y_true, y_pred, average='macro')
    f1_score_1 = metrics.f1_score(y_true, y_pred)
    f1_score_0 = metrics.f1_score(y_true, y_pred, pos_label=0)
    f1_score_avg = metrics.f1_score(y_true, y_pred, average='macro')
    return accuracy, precision_1, precision_0, precision_avg, recall_1, recall_0, recall_avg, f1_score_1, f1_score_0, \
           f1_score_avg, confusion_matrix


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=rest_port)
