async function trainModel(model, trainData, valData) {
    await model.fitDataset(trainData, {
      epochs: 20,
      validationData: valData,
      callbacks: tf.callbacks.earlyStopping({monitor: 'val_loss'})
    });
  }
  
  // Assuming you have trainData and valData as tf.data.Dataset objects
  await trainModel(model, trainData, valData);
  