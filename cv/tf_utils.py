import os

from tensorflow import keras


def train_model(x_train, y_train, x_val, y_val, model, params, logger):
    epochs = params['epochs']
    model_file = params['model_file']
    model_folder = params['model_folder']
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_file,
            save_weights_only=True,
            # save_best_only=False,
            save_best_only=True,
            mode='min'),
    ]
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    # Training model
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=1,  # 16,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )
    model.load_weights(model_file)
    return model
