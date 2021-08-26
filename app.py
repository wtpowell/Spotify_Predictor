from flask import Flask, render_template
import ml
from flask_cors import CORS


app = Flask(__name__)
app.secret_key = "lol"
@app.route('/')
def index():
    # if request.method == 'POST':
    #     danceability = request.form.get("myRange")
    #     energy = request.form.get("myRange2")
    #     tempo = request.form.get("myRange3")
        #session["danceability"] = danceability
        #return redirect(url_for('start', danceability=danceability, energy=energy, tempo=tempo))
        #return render_template('start.html', danceability=danceability, energy=energy, tempo=tempo)
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
    #ml.recommender.get_recommendations(danceability/100, energy/100, tempo/100, 1)
    #danceability= session.get('danceability', None)
    #return(danceability)
    # energy= request.args.get('energy', None)
    # tempo= request.args.get('tempo', None)
    #print (danceability)
    # return render_template('start.html')
    test = ml.test1(75, 90, 200)
    #test = ml.verygoodvariable
    return render_template('start.html', output = test)



    # file = open(r'/static/app.py', 'r').read()
    # return exec(file)

if __name__ == "__main__":
    app.run(debug=True)