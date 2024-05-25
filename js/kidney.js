import * as tf from '@tensorflow/tfjs';

function unetModel(inputShape) {
  const inputs = tf.input({shape: inputShape});
  
  function contractingBlock(input, filters) {
    const conv1 = tf.layers.conv2d({filters, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(input);
    const conv2 = tf.layers.conv2d({filters, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(conv1);
    return conv2;
  }

  function expansiveBlock(input, filters) {
    const upSample = tf.layers.upSampling2d().apply(input);
    const conv1 = tf.layers.conv2d({filters, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(upSample);
    const conv2 = tf.layers.conv2d({filters, kernelSize: 3, activation: 'relu', padding: 'same'}).apply(conv1);
    return conv2;
  }

  const enc1 = contractingBlock(inputs, 64);
  const pool1 = tf.layers.maxPooling2d({poolSize: 2}).apply(enc1);
  const enc2 = contractingBlock(pool1, 128);
  const pool2 = tf.layers.maxPooling2d({poolSize: 2}).apply(enc2);
  const enc3 = contractingBlock(pool2, 256);
  const pool3 = tf.layers.maxPooling2d({poolSize: 2}).apply(enc3);
  const enc4 = contractingBlock(pool3, 512);
  const pool4 = tf.layers.maxPooling2d({poolSize: 2}).apply(enc4);
  const bottleneck = contractingBlock(pool4, 1024);

  const dec4 = expansiveBlock(bottleneck, 512);
  const merge4 = tf.layers.concatenate().apply([dec4, enc4]);
  const dec3 = expansiveBlock(merge4, 256);
  const merge3 = tf.layers.concatenate().apply([dec3, enc3]);
  const dec2 = expansiveBlock(merge3, 128);
  const merge2 = tf.layers.concatenate().apply([dec2, enc2]);
  const dec1 = expansiveBlock(merge2, 64);
  const merge1 = tf.layers.concatenate().apply([dec1, enc1]);

  const outputs = tf.layers.conv2d({filters: 1, kernelSize: 1, activation: 'sigmoid'}).apply(merge1);

  const model = tf.model({inputs, outputs});
  model.compile({optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});
  
  return model;
}

// Usage
const model = unetModel([256, 256, 1]);
model.summary();
