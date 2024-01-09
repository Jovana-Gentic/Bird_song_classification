import librosa.display
from keras import Model, Input
from keras.layers import Conv1D, Dense, Dropout, SpatialDropout1D, GlobalAveragePooling1D

def create_model():
  inputs = Input((1026, 257), name = '0_Input_shape')

  x = Conv1D(128,8,2, activation='relu', padding='same', name = '1_Conv1D')(inputs)
  x = SpatialDropout1D(0.17, name = '2_SpatialDropout1D')(x)

  x = Conv1D(128,8,2, activation='relu', padding='same', name = '3_Conv1d')(x)
  x = SpatialDropout1D(0.17, name = '4_SpatialDropout1D')(x)

  x = Conv1D(128,8,2, activation='relu', padding='same', name = '5_Conv1d')(x)
  x = SpatialDropout1D(0.17, name = '6_SpatialDropout1D')(x)

  x = Conv1D(128,8,2, activation='relu', padding='same', name = '7_Conv1d')(x)
  x = SpatialDropout1D(0.17, name = '8_SpatialDropout1D')(x)

  x = GlobalAveragePooling1D(name = '9_Global_avg_pooling')(x)
  x = Dense(512, activation='relu', name='10_Dense')(x)
  x = Dropout(0.5, name='11_Dropout')(x)
  outputs = Dense(5, name='12_Dense')(x)
  model = Model(inputs=inputs, outputs=outputs, name = 'Bird_audio_classification_model')

  return model