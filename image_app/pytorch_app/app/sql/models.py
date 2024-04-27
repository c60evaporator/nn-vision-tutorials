from sqlalchemy import DateTime, Column, ForeignKey, Integer, String, Float, Date
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class SegEval(Base):
    __tablename__ = "seg_evaluations"
    evaluation_id = Column(Integer, primary_key=True, index=True)
    dataset = Column(String, index=True)
    image_set = Column(String, index=True)
    num_images = Column(Integer)
    model_name = Column(String, index=True)
    weight = Column(String)
    dataloader = Column(String)
    transform = Column(String)
    target_transform = Column(String)
    batch_size = Column(Integer)
    area_iou = Column(Float)
    mean_iou = Column(Float)
    label_names = Column(String)
    tps = Column(String)
    fps = Column(String)
    fns = Column(String)
    unions = Column(String)
    ious = Column(String)    
    elapsed_time = Column(Float)
    created_at = Column(DateTime)
    # Relationship
    image_evaluations = relationship('SegImageEval', back_populates='evaluation')

class SegImageEval(Base):
    __tablename__ = "seg_image_evaluations"
    image_evaluation_id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey('seg_evaluations.evaluation_id', ondelete='SET NULL'), nullable=False, index=True)
    img_path = Column(String)
    img_width = Column(Integer)
    img_height = Column(Integer)
    area_iou = Column(Float)
    mean_iou = Column(Float)
    created_at = Column(DateTime)
    # Relationship
    evaluation = relationship('SegEval', back_populates='image_evaluations')
    seg_evaluations = relationship('SegLabelImageEval', back_populates='image_evaluation')

class SegLabelImageEval(Base):
    __tablename__ = "seg_label_image_evaluations"
    label_image_evaluation_id = Column(Integer, primary_key=True, index=True)
    image_evaluation_id = Column(Integer, ForeignKey('seg_image_evaluations.image_evaluation_id', ondelete='SET NULL'), nullable=False, index=True)
    label_id = Column(Integer, index=True)
    label_name = Column(String, index=True)
    tp = Column(Integer)
    fp = Column(Integer)
    fn = Column(Integer)
    iou = Column(Float)
    created_at = Column(DateTime)
    # Relationship
    image_evaluation = relationship('SegImageEval', back_populates='seg_evaluations')