import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from flask import Flask, render_template, redirect, url_for
from infocrypt import infocrypt
from cybersentry_ai import cybersentry_ai
from lana_ai import lana_ai
from osint import osint
from portscanner import portscanner
from webseeker import webseeker
from filescanner import filescanner
from infosight_ai import infosight_ai
from snapspeak_ai import snapspeak_ai
from trueshot_ai import trueshot_ai
from enscan import enscan
from inkwell_ai import inkwell_ai

app = Flask(__name__, template_folder='static')

# Quick blueprint registration
blueprints = {
    '/infocrypt': infocrypt,
    '/cybersentry_ai': cybersentry_ai, 
    '/lana_ai': lana_ai,
    '/osint': osint,
    '/portscanner': portscanner,
    '/webseeker': webseeker,
    '/filescanner': filescanner,
    '/infosight_ai': infosight_ai,
    '/snapspeak_ai': snapspeak_ai,
    '/trueshot_ai': trueshot_ai,
    '/enscan': enscan,
    '/inkwell_ai': inkwell_ai
}

for prefix, blueprint in blueprints.items():
    app.register_blueprint(blueprint, url_prefix=prefix)

@app.route('/')
def login():
    return render_template('homepage.html')

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='127.0.0.1', port=5000)