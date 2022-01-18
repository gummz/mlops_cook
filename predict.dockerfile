docker run --name predict --rm \
  -v %cd%/trained_model.pt:/models/trained_model.pt \  # mount trained model file
  -v %cd%/data/example_images.npy:/example_images.npy \  # mount data we want to predict on
  predict:latest \
  ../../models/trained_model.pt \  # argument to script, path relative to script location in container
  ../../example_images.npy