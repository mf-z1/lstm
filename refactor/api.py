import torch
import argparse
import model
from flask import Flask, request, make_response, jsonify
from model import BiRNN
from datasets import text_field, label_field
from utils import transform_data
from flask import render_template

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# parser = argparse.ArgumentParser()
# parser.add_argument('--model-name', default='birnn',choices=['textcnn', 'birnn'], help='choose one model name for trainng')
# parser.add_argument('-lmd', '--load-model-dir', default= None, help='path for loadding model, default:None' )
# args = parser.parse_args()

# 获取模型名称
load_model_dir = 'models_storage/model_lstm.pt'
net = BiRNN()
net.load_state_dict(torch.load(load_model_dir))


# net = birnn()
# net.load_state_dict(torch.load('models_storage/model_birnn.pt'))
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/pre', methods=['GET', 'POST'])
def sentiemnt():
    # sentence = request.args.get('sentence')
    # if request.method == 'GET':
    #     return "1"
    if request.method == 'POST':
        sentence = request.form.get("sentent")
        # sentence = "你好"
    # if request.method == 'get':
        print(sentence)
        record = {'text': sentence}
        data, _ = transform_data(record, text_field, label_field)
        prediction = net(data).argmax(dim=1).item()
        if prediction == 0:
            result = '消极'
        else:
            result = '积极'
        return jsonify({'text': sentence, 'sentment': result})


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)
#

