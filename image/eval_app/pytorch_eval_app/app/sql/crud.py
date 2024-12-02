from typing import Dict, List, Any
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd

import sql.models as models

# Get evaluations
def get_evaluations(db: Session, model_name=None, convert_df: bool=False):
    if model_name is None:
        evaluations = db.query(models.SegEval).all()
    else:
        evaluations = db.query(models.SegEval).filter(models.SegEval.model_name == model_name).all()
    if convert_df:  # Return Pandas.DataFrame
        evaluations = [{k: v for k, v in evaluation.__dict__.items() if k != '_sa_instance_state'}
                       for evaluation in evaluations]
        return pd.DataFrame(evaluations)
    else:  # Return the SQLAlchemy object
        return evaluations

# Get image evaluations
def get_image_evaluations(db: Session, evaluation_id: int, convert_df: bool=False):
    image_evaluations = db.query(models.SegImageEval).filter(models.SegImageEval.evaluation_id == evaluation_id).all()
    if convert_df:  # Return Pandas.DataFrame
        image_evaluations = [{k: v for k, v in image_evaluation.__dict__.items() if k != '_sa_instance_state'}
                       for image_evaluation in image_evaluations]
        return pd.DataFrame(image_evaluations)
    else:  # Return the SQLAlchemy object
        return image_evaluations

# Create evaluation
def create_evaluation(db: Session, evaluation: List[Dict[str, Any]]):
    now = datetime.now()
    db_evaluation =  models.SegEval(**evaluation, created_at=now)
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    return db_evaluation

# Create image_evaluation
def create_image_evaluations(db: Session, image_evaluation: Dict[str, Any], evaluation_id: int, created_at: datetime):
    db_image_evaluation =  models.SegImageEval(**image_evaluation, 
            evaluation_id=evaluation_id, created_at=created_at)
    db.add(db_image_evaluation)
    db.commit()
    db.refresh(db_image_evaluation)
    return db_image_evaluation

# Create label_image_evaluations
def create_label_image_evaluations(db: Session, label_image_evaluations: List[Dict[str, Any]], image_evaluation_id: int, created_at: datetime):
    db_label_image_evaluations = [
        models.SegLabelImageEval(**label_image_evaluation, image_evaluation_id=image_evaluation_id, created_at=created_at)
        for label_image_evaluation in label_image_evaluations
    ]
    db.add_all(db_label_image_evaluations)
    db.commit()
    return db_label_image_evaluations

# Update evaluation
def update_evaluation(db: Session, evaluation: Dict[str, Any], evaluation_id: int):
    db.query(models.SegEval).filter_by(
        evaluation_id=evaluation_id).update(
        evaluation)
    db.commit()
    # Get updated data
    db_evaluation = db.get(models.SegEval, evaluation_id)
    return db_evaluation