import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

def get_db(database_dir):
    # Connect to the database
    os.makedirs(database_dir, exist_ok=True)
    db_url = f'sqlite:///{database_dir}/pytorch_app.db'
    engine = create_engine(db_url, connect_args={'check_same_thread': False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    import sql.models as models
    models.Base.metadata.create_all(bind=engine) # Create tables
    st.session_state['db'] = SessionLocal()
