from flask import Flask, request, json, jsonify
import base64
import numpy as np
import cv2
import pytesseract
from collections import namedtuple
import logging

DetectResult = namedtuple('Result', ('uid', 'has_text'))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# logging.getLogger('werkzeug').disabled = True


@app.route('/health_check', methods=['GET'])
def health_check():
    app.logger.debug('health_check')
    json = {
        'status': 200
    }
    return jsonify(json)


@app.route('/text_detect', methods=['POST'])
def text_detect():
    app.logger.debug('text_detect')
    posted = request.get_json()
    logging.getLogger('werkzeug').info(str(posted.keys()))
    base64_text = posted['image'].encode('utf-8')
    logging.getLogger('werkzeug').info(str(type(base64_text)))
    img_binary = base64.b64decode(base64_text)
    logging.getLogger('werkzeug').info(str(type(img_binary)))
    jpg = np.frombuffer(img_binary, dtype=np.uint8)
    logging.getLogger('werkzeug').info(str(type(jpg)))
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    logging.getLogger('werkzeug').info(str(type(img)))

    rectangles = posted['rectangles']
    config = ('-l eng+jpn --oem 1 --psm 3')
    res_rectangles = []
    for rect in rectangles:
        x, y, width, height, uid = rect
        clip_img = img[y:y+height, x:x+width]
        # Tesseract の呼び出し
        text = pytesseract.image_to_string(clip_img, config=config)
        if text == '':
            res_rectangles.append(DetectResult(uid, False))
        else:
            res_rectangles.append(DetectResult(uid, True))

    res_json = json.dumps(res_rectangles)
    app.logger.debug(res_json)
    return jsonify(json.loads(res_json))


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def error_handler(error):
    response = jsonify({
                          "error": {
                          "type": error.name,
                          "message": error.description
                          }
                      })
    return response, error.code
