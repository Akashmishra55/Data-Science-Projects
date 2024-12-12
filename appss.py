from flask import Flask,render_template,request
import pickle

app=Flask(__name__)

sv=pickle.load(open("sv.pkl","rb"))
tf_vect=pickle.load(open("tf_vect.pkl","rb"))


@app.route("/")
def index():

   return render_template("index.html")
@app.route("/detect",methods=["POST"])
def detect_plagiarism():
   input_text=request.form["text"]
   vectrized_text=tf_vect.transform([input_text])
   result=sv.predict(vectrized_text)
   result="Plagiarism Detected" if result[0]==1 else "No Plagiarism"
   return render_template("index.html",result=result)

if __name__=='__main__':
   app.run(debug=True)

