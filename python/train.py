# Assuming you have train_images, train_labels, val_images, val_labels
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=20, batch_size=32)
