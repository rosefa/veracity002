class CustomConstraint(Constraint):
    n=5
    def __init__(self,k,s):
      self.k = k
      self.s = s
      np.random.rand2 = lambda *args, dtype=np.float32: np.random.rand(*args).astype(dtype)
      f1 = np.random.rand2(s,s)
      custom_weights = np.array((f1, f1, f1))
      f2 = custom_weights.transpose(1, 2, 0)
      custom_weights = np.tile(f2, (k, 1, 1))
      T2 = np.reshape(custom_weights,(k,s,s,3))
      custom_weights = T2.transpose(1, 2, 3, 0)
      self.custom_weights = tf.Variable(custom_weights)
    def __call__(self, weights):
      weights = self.custom_weights
      row_index = self.s//2
      col_index = self.s//2
      new_value = 0
      weights[row_index,col_index,:,:].assign(new_value)
      som = tf.keras.backend.sum(weights)
      sum_without_center1 = 1/som
      newMatrix = weights*sum_without_center1
      weights.assign(newMatrix)
      new_value = -1
      weights[row_index,col_index,:,:].assign(new_value)
      return weights

class FakeMultimodal:
    def __init__(self, bert_layer, max_seq_length=512, epochs=100, batch_size=70):
      # BERT and Tokenization params
      self.bert_layer = bert_layer
      self.max_seq_length = max_seq_length        
      vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
      do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
      self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
      
      # Learning control params
      self.epochs = epochs
      self.batch_size = batch_size
      self.models = []
      self.scores = {}
    
  def bert_encode(self, texts):
    all_tokens = []
    all_masks = []
    all_segments = []
    for text in texts:
      text = self.stokenizer.tokenize(text)
      text = text[:self.max_seq_length-2]
      input_sequence = ["[CLS]"] + text + ["[SEP]"]
      pad_len = self.max_seq_length-len(input_sequence)
      tokens = self.tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
      pad_masks = [1] * len(input_sequence) + [0] * pad_len
      segment_ids = [0] * self.max_seq_length

      all_tokens.append(tokens)
      all_masks.append(pad_masks)
      all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def fake_virtual(self):
  input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_word_ids")
  input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name="input_mask")
  segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name="segment_ids")
  pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
  clf_output = sequence_output[:, 0, :]
  
  model1 = Bidirectional(LSTM(32))(clf_output)
  model1 = Dense(64, activation='softmax')(model1)

  input2 = Input(shape=(224,224,3))
  model2 = Conv2D(filters=3, kernel_size=5,kernel_constraint=CustomConstraint(3,5), padding='same', use_bias=False)(input2)
  model2 = Conv2D(filters=16,kernel_size=3, padding='same',use_bias=False)(model2)
  model2 = BatchNormalization(axis=3, scale=False)(model2)
  model2 = Activation('relu')(model2)
  model2 = Conv2D(filters=32,kernel_size=3, padding='same',use_bias=False)(model2)
  model2 = BatchNormalization(axis=3, scale=False)(model2)
  model2 = Activation('relu')(model2)
  model2 = GlobalAveragePooling2D()(model2)
  model2 = Dense(64, activation='relu')(model2)

  outFinal = tf.keras.layers.Add()([model1, model2])
  final_model_output = Dense(2, activation='softmax')(outFinal)
  #input1=[input_word_ids, input_mask, segment_ids]
  final_model = Model(inputs=[input_word_ids, input_mask, segment_ids, input2], outputs=final_model_output)
  #final_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy", f1_m])
  final_model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='rappel')])
  return final_model

def train(self, X):
  result = []
  scores_loss = []
  scores_acc = []
  scores_pre = []
  scores_rap = []
  fold_var = 1
  for train_indices, val_indices in kf.split(self.X):
    print(fold_var)
    train = self.X.iloc[train_indices]
    val = self.X.iloc[val_indices]
    label = preprocessing.LabelEncoder()
    
    trainText = self.bert_encode(train['text'], tokenizer, max_len=max_len)
    trainImage = train['image']
    trainImage = trainImage.to_numpy()
    # Conversion
    trainImage = np.array([val for val in trainImage])
    trainLabel = label.fit_transform(train['label'])
    trainLabel = to_categorical(trainLabel)
    labels = label.classes_
    # print(labels)
    # print(trainLabel)
    valText = self.bert_encode(val['text'], tokenizer, max_len=max_len)
    valImage = val['image']
    valImage = valImage.to_numpy()
    valImage = np.array([val for val in valImage])
  
    valLabel = label.fit_transform(val['label'])
    valLabel = to_categorical(valLabel)
    valLabel = valLabel
  
    testText = bert_encode(dataImageTextTest['text'], tokenizer, max_len=max_len)
    testImage = dataImageTextTest['image']
    testImage = testImage.to_numpy()
    testImage = np.array([val for val in testImage])
  
    testLabel = label.fit_transform(dataImageTextTest['label'])
    testLabel = to_categorical(testLabel)
    testLabel = testLabel
    
    file_path = save_dir+'model_'+str(fold_var)+".hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=8)
    callbacks_list = [checkpoint,earlystopping]
    #print(type(trainText),type(trainImage),type(trainLabel))
  
    hist = model.fit(x=[trainText,trainImage],y=trainLabel,epochs=30,batch_size=70,validation_data=([valText, valImage], valLabel), callbacks=callbacks_list, verbose=1)
    model.load_weights(file_path)
    score = model.evaluate([valText, valImage],valLabel, verbose=0)
    scores_loss.append(score[0])
    scores_acc.append(score[1])
    scores_pre.append(score[2])
    scores_rap.append(score[3])
    result.append(model.predict([testText,testImage]))
    '''
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plot_accuracy(epochs, accuracy,val_accuracy,fold_var)
  
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plot_loss(epochs, loss,val_loss,fold_var)
  
    fper, tper, thresholds = roc_curve(testLabel, y_pred)
    plot_roc_curve(fper, tper,fold_var)
    '''
    tf.keras.backend.clear_session()
    fold_var += 1
value_min = min(scores_loss)
value_index = scores_loss.index(value_min)
model.load_weights(save_dir+'model_'+str(value_index)+".hdf5")
best_model = model
best_model.save_weights('model_weights.h5')
best_model.save('model_keras.h5')
