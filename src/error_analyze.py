def error_analzyze(texts, y_label, y_pred, probs):
    errors = []

    for text, label, pred, prob in zip(texts, y_label, y_pred, probs):
        if label != pred:
            errors.append({"text": text, "label": label, "pred": pred, "prob": prob})
    return errors
