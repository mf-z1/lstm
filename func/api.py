import torch

from flask import Flask, request, make_response, jsonify
from model import BiRNN
from datasets import text_field, label_field
from utils import transform_data
from flask import render_template

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False

# 获取模型名称
load_model_dir = 'models_storage/model_lstm.pt'
net = BiRNN()
net.load_state_dict(torch.load(load_model_dir))


@app.route('/')
def index():
    index_html = 'index.html'
    return render_template(index_html)


@app.route('/pre', methods=['GET', 'POST'])
def sentiments():
    if request.method == 'POST':
        sentiment = request.form.get("sentiment")

        record = {'text': sentiment}
        data, _ = transform_data(record, text_field, label_field)
        prediction = net(data).argmax(dim=1).item()
        if prediction == 0:
            result = '消极'
        else:
            result = '积极'
        return jsonify({'text': sentiment, 'sentment': result})


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", debug=True)
#
