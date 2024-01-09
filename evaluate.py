import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import seaborn as sns

from data import make_dataset,split_data
_, bird_filepaths_val, _, bird_labels_val = split_data()
dataset_val = make_dataset(bird_labels_val, bird_filepaths_val, shuffle=False)

checkpoint_filepath = 'model_checkpoint/topmodel'

model_eval = tf.keras.models.load_model(checkpoint_filepath)
print(model_eval.evaluate(dataset_val, return_dict=True))

y_logits = model_eval.predict(dataset_val)
y_prob = tf.nn.softmax(y_logits, axis=1)
y_pred = tf.argmax(y_logits, axis=1)
y_true = tf.concat(list(dataset_val.map(lambda s,lab: lab)), axis=0)

avg_precision = average_precision_score(y_true, y_prob, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
auc = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
print(f"AP: {avg_precision} | F1: {f1} | Precision: {precision} | Recall: {recall} | AUC: {auc}")

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=['bewickii','polyglottos','migratorius','melodia','cardinalis'],
            yticklabels=['bewickii','polyglottos','migratorius','melodia','cardinalis'],
            annot=True, fmt='g', cmap = 'Blues')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()