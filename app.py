from flask import Flask, render_template
import ml


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process')
def process():
    return render_template('process.html')

@app.route('/visual1')
def visual1():
    return render_template('visual1.html')

@app.route('/visual2')
def visual2():
    return render_template('visual2.html')

@app.route('/start')
def button():
    test = ml.verygoodvariable
    return render_template('start.html', output = test)



    # file = open(r'/static/app.py', 'r').read()
    # return exec(file)

if __name__ == "__main__":
    app.run(port = 8000, debug=True)