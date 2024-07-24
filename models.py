from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    img_path = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return '<DetectionHistory %r>' % self.id
