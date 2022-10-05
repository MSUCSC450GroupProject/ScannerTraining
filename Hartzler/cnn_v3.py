import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
### Also need to: pip install pillow
base_path = "./word_cloud_data/"
SHAPE = (224,224,3)
batch_size = 32

print(tf.config.list_physical_devices('GPU'))

#============================================================================================================================
def generate_model(input_sz, ker_size, num_classes):
    inputs = Input(shape=input_sz)
    ######### Feature Extractors ############
    L1   = Conv2D(filters = 64, kernel_size = ker_size, activation = 'relu')(inputs) # => Outputs 64 matrixices of ker_size x ker_size
    L2   = MaxPooling2D(pool_size = 2, padding = 'same')(L1) # => Downsamples the input along its (height and width) by taking the maximum value over an input window (of size defined by pool_size)
    
    L3   = Conv2D(filters = 64, kernel_size = ker_size, activation = 'relu', padding='same')(L2)  
    L4   = MaxPooling2D(pool_size = 2, padding = 'same')(L3)
    
    
    L5   = Conv2D(filters = 64, kernel_size = ker_size, activation = 'relu', padding='same')(L4)
    L6   = MaxPooling2D(pool_size = 2, padding = 'same')(L5)
      
    L7   = Conv2D(filters = 64, kernel_size = ker_size, activation = 'relu', padding = 'same')(L6)
    L8   = MaxPooling2D(pool_size = 2, padding = 'same')(L7)
    ######### END Feature Extractors ############ 

    L9   = Flatten()(L8)
    L10  = Dense(80, activation = 'relu')(L9)
    
    L11  = Dropout(rate=0.17)(L10)
    L12  = Dense(num_classes, activation='softmax')(L11)
        
    cnn_model = Model(inputs=inputs, outputs=L12)
        
    cnn_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    cnn_model.summary()
        
    return cnn_model
#=============================================================================================================================
def  Print_Res(tslabels, res):
    cm = confusion_matrix(tslabels, res)
    print(cm)
    print("\n")        
    pr = precision_score(tslabels, res, average=None)
    print("Precision:", pr)
    rc = recall_score(tslabels, res, average=None)
    print("   Recall:", rc)
    f1 = f1_score(tslabels, res, average=None)
    print("       F1:", f1)
    ac = accuracy_score(tslabels, res)
    print(" Accuracy:", ac)
        
        
    print("\n")        
    pr = precision_score(tslabels, res, average='macro')
    print("Precision:", pr)
    rc = recall_score(tslabels, res, average='macro')
    print("   Recall:", rc)
    
    f1 = f1_score(tslabels, res, average='macro')
    print("       F1:", f1)
#=============================================================================================================================

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen  = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
            base_path + 'train/',
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = True,
            seed = 33,
            classes = ['non_sarcastic','sarcastic'])

test_generator = test_datagen.flow_from_directory(
            base_path + 'test/',
            target_size = (SHAPE[0], SHAPE[1]),
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False,
            seed = 33,
            classes = ['non_sarcastic','sarcastic'])


model = generate_model(SHAPE, 5, 2)
model.fit(train_generator, steps_per_epoch=train_generator.samples/train_generator.batch_size, epochs=400)

test_num = test_generator.samples

predict = model.predict(test_generator)


pred_test = np.argmax(predict, axis=1)


label_test = []
for i in range((test_num // test_generator.batch_size)+1):
     X,y = test_generator.next()
     label_test.append(y)

label_test = np.argmax(np.vstack(label_test), axis=1)


Print_Res(label_test, pred_test)