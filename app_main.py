# -*- coding:utf-8 -*-

from flask import Flask, request, jsonify
import traceback
import logging
import time
import logging
import time

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)

app = Flask("my-app")




@app.route('/post_server', methods=['POST'])
def post_server():
    try:
        aaa = time.time()
        data = request.get_json()
        content = data['content']
        type = data["type"]

        logging.info('type = %s , content=%s', content, type)

        bbb = time.time()
        logging.info("spend time {}".format(bbb - aaa))
        return str(content)
    except:
        logging.info('****预测服务出错****')
        time.sleep(1)
        return str("*****predict something wrong*****")


@app.route('/get_server', methods=["GET"])
def get_server():
    print("model already started !")
    return "1"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
