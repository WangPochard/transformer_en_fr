import time

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow import keras
import re
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer

print(tf.test.is_gpu_available())
path = 'E:/machinelearning/translate'
path_ = 'E:/transformer'
log_dir = os.path.join(path_, 'logs')
checkpoint_path = os.path.join(path_, 'checkpoints')

# gpu setting
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

# 使用 100% 的 GPU 記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 TensorFlow Session
tf.compat.v1.keras.backend.set_session(sess)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights

class multi_head_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(multi_head_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  # 用來確認維度d_model是否可以均分成num_heads個 depth(注意張量:通常會與output的維度一樣)維度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # 給每個head的字詞新的repr.(表示)維度
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # mask = tf.squeeze(mask, axis=1) # 把mask維度降到3維

        scaled_attention, attention_weight = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weight


def feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='ReLU'),
        tf.keras.layers.Dense(d_model)
    ])


def create_padding_mask(seq):
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mhn = multi_head_attention(d_model, num_heads)  # multi head attention
        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attn = self.mhn(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout1(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # seq_len, seq_len


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = multi_head_attention(d_model, num_heads)
        self.mha2 = multi_head_attention(d_model, num_heads)
        self.ffn = feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, combined_mask, padding_mask):
        attn1, attn_weights1 = self.mha1(x, x, x, combined_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights2 = self.mha2(out1, encoder_output,
                                         encoder_output, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights1, attn_weights2


# position encoding(PE) (tutorial by tensorflow)
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2) / np.float32(d_model)))
    return pos * angle_rates


def PE(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sin (2i)
    sin_p = np.sin(angle_rads[:, 0::2])  # 偶數索引
    # cos (2i+1)
    cos_p = np.cos(angle_rads[:, 1::2])  # 奇數索引

    pos_encoding = np.concatenate([sin_p, cos_p], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, source_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(source_vocab_size, d_model)
        self.pos_encoding = PE(source_vocab_size, self.d_model)

        self.encoding_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        input_seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]

        x = self.dropout(x, training=training)

        for i, encoding_layer in enumerate(self.encoding_layers):
            x = encoding_layer(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PE(target_vocab_size, self.d_model)
        self.decoding_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                                for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, combined_mask, padding_mask):
        tar_seq_len = tf.shape(x)[1]
        attention_weights = {}  # 存放Docoder layer的 attention權重

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tar_seq_len, :]

        x = self.dropout(x, training=training)

        for i, decoding_layer in enumerate(self.decoding_layers):
            x, block1, block2 = decoding_layer(x, encoder_output, training,
                                               combined_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, source_vocab_size,
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               source_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, en_padding_mask,
             de_padding_mask, combined_mask):
        encoder_output = self.encoder(inp, training, en_padding_mask)

        decoder_output, attention_weights = self.decoder(tar,
                                                         encoder_output, training,
                                                         combined_mask, de_padding_mask)

        final_output = self.final_layer(decoder_output)

        return final_output, attention_weights


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)

    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

#-----------------------------------------------------------------
# load data
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv(path+"/eng_-french.csv")
data = shuffle(data)[:]

#data processing
english_text = data[data.columns[0]]
french_text = data[data.columns[1]]

english = []
french = []
for i in english_text.index:
    text = english_text[i].lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    english.append(text)

for i in french_text.index:
    ftext = french_text[i].lower()
    ftext = (re.sub("[^a-zA-Z' àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]", ' ', ftext))
    french.append(ftext)

cleaned_data = pd.DataFrame({'english': english, 'french': french})

# Vocabulary of English
all_eng_words = set()
for i in english:
    for j in i.split():
        all_eng_words.add(j)

# vocabulary of french
all_fre_words = set()
for i in french:
    for j in i.split():
        all_fre_words.add(j)

# maxlen of the source sequence
max_length_src = 0
for i in english:
    a = len(i.split())
    if a > max_length_src:
        max_length_src = a

# maxlen of the target sequence
max_length_tar = 0
for j in french:
    b = len(j.split())
    if b > max_length_tar:
        max_length_tar = b

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_fre_words))

# Calculate Vocab size for both source and targe
num_encoder_tokens = len(all_eng_words)+2
num_decoder_tokens = len(all_fre_words)+2

# indexs for input and target sequences
input_index = dict([(words, i) for i, words in enumerate(input_words)])
target_index = dict([(word, i) for i, word in enumerate(target_words)])

reverse_input_index = dict((i, word) for word, i in input_index.items())
reverse_target_index = dict((i, word) for word, i in target_index.items())

#隨機排序data資料
cleaned_data = shuffle(cleaned_data)

#split training & validation data
val_n = round(len(cleaned_data)/10)

train_examples = cleaned_data[:len(cleaned_data)-val_n]
val_examples = cleaned_data[len(cleaned_data)-val_n:]

#let's make data invert to token mode
### english data
x_tokens = Tokenizer()
x_tokens.fit_on_texts(input_words)
en_token = x_tokens.texts_to_sequences(cleaned_data['english'])
# add start token & end token in english corpus
en_token = [[num_encoder_tokens-2] + sentence + [num_encoder_tokens-1] for sentence in en_token]
#english token in training data
en_t_token = en_token[:len(cleaned_data)-val_n]
#english token in validation data
en_v_token = en_token[len(cleaned_data)-val_n:]

### french data
y_tokens = Tokenizer()
y_tokens.fit_on_texts(target_words)
fn_token = y_tokens.texts_to_sequences(cleaned_data['french'])
# add start token & end token in french corpus
fn_token = [[num_decoder_tokens-2] + sentence + [num_decoder_tokens-1] for sentence in fn_token]
#french token in training data
fn_t_token = fn_token[:len(cleaned_data)-val_n]
#french token in validation data
fn_v_token = fn_token[len(cleaned_data)-val_n:]



# Removing too long sentences
MAX_LENGTH = 20
idx_to_remove = [count for count, sent in enumerate(en_t_token) if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del en_t_token[idx]
    del fn_t_token[idx]
idx_to_remove = [count for count, sent in enumerate(fn_t_token) if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del en_t_token[idx]
    del fn_t_token[idx]

idx_to_remove = [count for count, sent in enumerate(en_v_token) if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del en_v_token[idx]
    del fn_v_token[idx]
idx_to_remove = [count for count, sent in enumerate(fn_v_token) if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del en_v_token[idx]
    del fn_v_token[idx]

#生成padding data
with tf.device('/cpu:0'):
    train_data = tf.constant(train_examples)
    val_data = tf.constant(val_examples)
    en_t, fn_t = next(iter(train_data))

    #padding training data
    en_pad_train = tf.keras.preprocessing.sequence.pad_sequences(en_t_token, padding='post', maxlen = MAX_LENGTH)
    en_pad_train = tf.constant(en_pad_train)
    fn_pad_train = tf.keras.preprocessing.sequence.pad_sequences(fn_t_token, padding='post', maxlen = MAX_LENGTH)
    fn_pad_train = tf.constant(fn_pad_train)
    #padding validation data
    en_pad_val = tf.keras.preprocessing.sequence.pad_sequences(en_v_token, padding='post', maxlen = MAX_LENGTH)
    en_pad_val = tf.constant(en_pad_val)
    fn_pad_val = tf.keras.preprocessing.sequence.pad_sequences(fn_v_token, padding='post', maxlen = MAX_LENGTH)
    fn_pad_val = tf.constant(fn_pad_val)
    print(len(en_pad_train))
    #print(en_pad_train[:16,:])

BATCH_SIZE = 128
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((en_pad_train, fn_pad_train))

dataset = dataset.cache() # To increase speed
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) #Data is prefetched to increase speed
#------------------------------------------------------------------data precessing 完成


# 觀察loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    # 將序列中不等於0的位置是為1
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#TensorBoard
#設置keras metric (loss func & Accuracy)
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
name = 'train_accuracy')

num_layers = 6
d_model = 256
dff = 512
num_heads = 8
dropout_rate = 0.1


# Optimizer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          num_encoder_tokens, num_decoder_tokens, dropout_rate)

# checkpoints setting
run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
# 5個 checkpoints
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
else:
    last_epoch = 0
    print("沒找到 checkpoint，從頭訓練。")

# inp,tar皆是padding後的data
@tf.function
def train_step(inp, tar):
    #print(tar)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_pad_mask, combined_mask, dec_pad_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_pad_mask,
                                     dec_pad_mask, combined_mask)

        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

#--------------------------------------
EPOCHS = 30
print(f"此超參數組合的 Transformer 已經訓練 {last_epoch} epochs。")
print(f"剩餘 epochs：{min(0, last_epoch - EPOCHS)}")


summary_writer = tf.summary.create_file_writer(log_dir)
with tf.device('/gpu:0'):
    loss_ytrain = []
    for epoch in range(last_epoch, EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        #for step_idx in range(len(en_pad_train)//64):
        for (batch, (enc_inputs, targets)) in enumerate(dataset):
            train_step(enc_inputs, targets)

        with summary_writer.as_default():
            train_loss_tensor = tf.constant(train_loss.result())
            train_accuracy_tensor = tf.constant(train_accuracy.result())
            tf.summary.scalar("train_loss", train_loss_tensor, step=epoch + 1)
            tf.summary.scalar("train_acc", train_accuracy_tensor, step=epoch + 1)

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(),
                                                            train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


        loss_ytrain.append(train_loss.result())
    print("loss_ytrain:",loss_ytrain)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    loss_x = np.arange(1,EPOCHS+1)
    loss_ytrain = np.asarray(loss_ytrain)
    plt.plot(loss_x,loss_ytrain)
    plt.savefig('loss_plot.png')
    plt.show()

#----------------------------transformer evaluate------------------------
def evaluate(inp_sentence):
    inp_sentence = [num_encoder_tokens-2] + x_tokens.texts_to_sequences([inp_sentence])[0] + [num_encoder_tokens-1]
    encoder_input = tf.expand_dims(inp_sentence, axis = 0)

    output = tf.expand_dims([num_decoder_tokens-2], axis = 0)
    print("\nencoder_input:\n", encoder_input)
    for i in range(MAX_LENGTH):
        enc_pad_mask, combined_mask, dec_pad_mask = create_masks(encoder_input, output)
        predictions , attn_weights = transformer(encoder_input, output, False, enc_pad_mask,
                                     dec_pad_mask, combined_mask)
                #predictions's shape: (1, seq_length, num_decoder_token)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis = -1), tf.int32)
        if tf.equal(predicted_id, num_decoder_tokens-1):
            return tf.squeeze(output, axis = 0)
        output = tf.concat([output, predicted_id], axis = -1)
    print('-'*10,"\noutput_squeeze:\n", tf.squeeze(output, axis = 0))

    return tf.squeeze(output, axis = 0)
def translate(sentence):
    output_ = evaluate(sentence).numpy()
    print("\n\noutput:\n",output_)
    for i in output_:
        print("decode_output:",i)
    predicted_sentence = y_tokens.sequences_to_texts(
        [[i] for i in output_ if i < (num_decoder_tokens-2)]
    )
    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(predicted_sentence))
#------------------------------------------------------------------------
real = np.asarray(val_examples['french'])
val_examples = np.asarray(val_examples['english'])
translate(str(val_examples[0]))
print("Real French:", real[0])

translate(str(val_examples[5]))
print("Real French:", real[5])