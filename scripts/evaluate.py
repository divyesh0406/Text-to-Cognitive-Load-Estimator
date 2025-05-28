from sklearn.metrics import classification_report

def evaluate(trainer, dataset):
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    print(classification_report(labels, preds, target_names=["Low", "Medium", "High"]))
