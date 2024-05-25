import * as tf from '../@tensorflow/tfjs';
import daikon from '../daikon';

// Utility function for converting DICOM to a tensor
async function dicomToTensor(dicomFile) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const buffer = event.target.result;
      const data = new DataView(buffer);
      const image = daikon.Series.parseImage(data);

      if (image === null) {
        reject(new Error('Could not parse DICOM image.'));
        return;
      }

      const rows = image.getRows();
      const cols = image.getCols();
      const pixelData = image.getInterpretedData(true);

      const imageData = new Float32Array(rows * cols);
      for (let i = 0; i < rows * cols; i++) {
        imageData[i] = pixelData[i];
      }

      const tensor = tf.tensor3d(imageData, [rows, cols, 1]);
      resolve(tensor);
    };
    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsArrayBuffer(dicomFile);
  });
}

// Utility function for contracting block in U-Net
function contractingBlock(input, filters) {
  const conv1 = tf.layers.conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(input);
  const conv2 = tf.layers.conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(conv1);
  return conv2;
}

// Utility function for expansive block in U-Net
function expansiveBlock(input, filters) {
  const upSample = tf.layers.upSampling2d().apply(input);
  const conv1 = tf.layers.conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(upSample);
  const conv2 = tf.layers.conv2d({ filters, kernelSize: 3, activation: 'relu', padding: 'same' }).apply(conv1);
  return conv2;
}

// Brain Tumor Segmentation Model
function unetModel(inputShape) {
  const inputs = tf.input({ shape: inputShape });

  const enc1 = contractingBlock(inputs, 64);
  const pool1 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(enc1);
  const enc2 = contractingBlock(pool1, 128);
  const pool2 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(enc2);
  const enc3 = contractingBlock(pool2, 256);
  const pool3 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(enc3);
  const enc4 = contractingBlock(pool3, 512);
  const pool4 = tf.layers.maxPooling2d({ poolSize: 2 }).apply(enc4);
  const bottleneck = contractingBlock(pool4, 1024);

  const dec4 = expansiveBlock(bottleneck, 512);
  const merge4 = tf.layers.concatenate().apply([dec4, enc4]);
  const dec3 = expansiveBlock(merge4, 256);
  const merge3 = tf.layers.concatenate().apply([dec3, enc3]);
  const dec2 = expansiveBlock(merge3, 128);
  const merge2 = tf.layers.concatenate().apply([dec2, enc2]);
  const dec1 = expansiveBlock(merge2, 64);
  const merge1 = tf.layers.concatenate().apply([dec1, enc1]);

  const outputs = tf.layers.conv2d({ filters: 1, kernelSize: 1, activation: 'sigmoid' }).apply(merge1);

  const model = tf.model({ inputs, outputs });
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

  return model;
}

async function trainModel(model, trainData, valData) {
  await model.fitDataset(trainData, {
    epochs: 20,
    validationData: valData,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss' }),
  });
}

document.getElementById('trainButton').addEventListener('click', async () => {
  const dicomInput = document.getElementById('dicomInput');
  const dicomFiles = dicomInput.files;

  if (dicomFiles.length === 0) {
    alert('Please select DICOM files to train the model.');
    return;
  }

  const tensors = [];
  for (let i = 0; i < dicomFiles.length; i++) {
    try {
      const tensor = await dicomToTensor(dicomFiles[i]);
      tensors.push({ xs: tensor, ys: tensor }); // Example for autoencoder-like training
    } catch (error) {
      console.error('Error processing DICOM file:', dicomFiles[i].name, error);
    }
  }

  const trainData = tf.data.array(tensors);
  const valData = tf.data.array(tensors); // Replace with actual validation data

  const model = unetModel([256, 256, 1]);
  model.summary();

  document.getElementById('output').innerText = 'Training model...';

  await trainModel(model, trainData, valData);

  document.getElementById('output').innerText = 'Training completed.';
});
