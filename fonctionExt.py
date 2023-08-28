def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def fake_virtual(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    #input1 = Input(shape=(100,))
    #embedding_layer = Embedding(len(word_index)+1,100,embeddings_initializer=keras.initializers.Constant(embedding_matrix),input_length=100,trainable=False)(input1)
    #model1 = Bidirectional(LSTM(32))(embedding_layer)
    model1 = Bidirectional(LSTM(32))(sequence_output)
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

#K.clear_session()

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

def plot_roc_curve(fper, tper,fold_var):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig('courbes_ROC_'+str(fold_var)+'.png')

def plot_loss(epochs, loss,val_loss):
  plt.plot(epochs, loss, 'b', label='loss')
  plt.plot(epochs, val_loss, 'r', label='Val loss')
  plt.title('Loss and Val_Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('courbes_loss.png')

def plot_accuracy(epochs, accuracy,val_accuracy):
  plt.plot(epochs, accuracy, 'b', label='Accuracy')
  plt.plot(epochs, val_accuracy, 'r', label='Val Accuracy')
  plt.title('Accuracy and Val Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig('courbes_Accuracy.png')
