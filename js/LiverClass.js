import * as tf from '@tensorflow/tfjs';

async function liverClassifierModel() {
  const baseModel = await tf.loadLayersModel('https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4', {fromTFHub: true});
  const inputs = tf.input({shape: [224, 224, 3]});
  const x = baseModel.apply(inputs, {training: false});
  const x = tf.layers.dense({units: 512, activation: 'relu'}).apply(x);
  const predictions = tf.layers.dense({units: 1, activation: 'sigmoid'}).apply(x);
  const model = tf.model({inputs, outputs: predictions});

  model.compile({optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});
  
  return model;
}

// Usage
const model = await liverClassifierModel();
model.summary();
