# -*- coding: utf-8 -*-

def get_train_test(df, attribute):
  train_data = df[df['type']=='train']
  X_train = train_data[attribute].map(lambda x: x.split())
  y_train = train_data['label']

  test_data = df[df['type']=='test']
  X_test = test_data[attribute].map(lambda x: x.split())
  y_test = test_data['label']

  return X_train, y_train, X_test, y_test

def text_tokenizer(X_train, X_test, max_length):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_train)
  X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)
  X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_length)

  return X_train, X_test, tokenizer

def get_embedding_matrix(tokenizer, w2v_model, embedding_dim):
  vocab_size = len(tokenizer.word_index)+1
  embedding_matrix = np.zeros((vocab_size, embedding_dim))
  
  for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
      embedding_matrix[i] = w2v_model.wv[word]
  
  return embedding_matrix, vocab_size


def plot_acc(train_acc, valid_acc, title):
  epochs = range(len(train_acc))
  plt.figure(figsize=(8,6))
  plt.plot(epochs, train_acc, 'o-', label='training accuracy')
  plt.plot(epochs, valid_acc, 'o-', label='validation accuracy')
  plt.xlabel('epoch', fontsize=14)
  plt.ylabel('accuracy', fontsize=14)
  plt.title(title, fontsize=15)
  plt.legend(loc='lower right')
  plt.grid()
  plt.show()

  return None

def plot_loss(train_loss, valid_loss, title):
  epochs = range(len(train_loss))
  plt.figure(figsize=(8,6))
  plt.plot(epochs, train_loss, 'o-', label='training loss')
  plt.plot(epochs, valid_loss, 'o-', label='validation loss')
  plt.xlabel('epoch', fontsize=14)
  plt.ylabel('loss', fontsize=14)
  plt.title(title, fontsize=15)
  plt.legend()
  plt.grid()
  plt.show()

  return None

def plot_roc_curve(fpr, tpr):
  auc_value = auc(fpr, tpr)
  print("AUC : ", auc_value)
  
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(auc_value))
  plt.plot([0, 1], [0, 1], 'k--', label='y=x')
  plt.xlabel('False positive rate', fontsize=12)
  plt.ylabel('True positive rate', fontsize=12)
  plt.title('ROC curve', fontsize=14)
  plt.legend(loc='best', fontsize=12)
  #plt.savefig('A_ROC/0123-4val.jpg')
  plt.grid()
  plt.show()

  return


def compute_acc(pred, label, thresholds):
  pred = (pred > thresholds).astype(int)
  label = np.array(label).reshape(-1, 1)
  acc = np.sum((pred == label).astype(int)) / len(y_test)

  return acc


def get_best_thresholds(model, pred, label):
  fpr, tpr, thresholds = roc_curve(label, pred)
  best_thresholds = thresholds[len(thresholds)-np.argmax(tpr-fpr)]
  
  return fpr, tpr, best_thresholds


