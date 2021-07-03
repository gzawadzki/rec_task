from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os

#Init app
app= Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

#Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+ os.path.join(basedir,'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#Init db
db = SQLAlchemy(app)

#init marshmallow
ma = Marshmallow(app)


# Word Class/Model
class Word(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word= db.Column(db.String(100))

    def __init__(self,word):
        self.word=word

# Word Schema
class WordSchema(ma.Schema):
    class Meta:
        fields = ('id','word')

# Init Schema
word_schema = WordSchema()
words_schema= WordSchema(many = True)


# create a word
@app.route('/word', methods=['POST'])
def add_word():
    word = request.json['word']
    new_word = Word(word)
    db.session.add(new_word)
    db.session.commit()

    return word_schema.jsonify(new_word)

#get all words
@app.route('/words',methods=['GET'])
def get_words():
    all_words = Word.query.all()
    result = words_schema.dump(all_words)
    return jsonify(result)

#delete word 
@app.route('/words/<id>',methods=['DELETE'])
def delete_word(id):
    word = Word.query.get(id)
    db.session.delete(word)
    db.session.commit()
    
    return word_schema.jsonify(word)



#run server

if __name__ == '__main__':
    app.run(debug=True)
