async function loadModel() {
    const model = await tf.loadLayersModel("./model.json");
    return model;
  }
  
  async function classifyImage(imageElement) {
    const model = await loadModel();
  
    const inputImage = tf.browser.fromPixels(imageElement).resizeNearestNeighbor([28, 28]).toFloat();
    const grayImage = inputImage.mean(2).expandDims(-1);
    const normalizedImage = grayImage.div(tf.scalar(255));
    const inputTensor = normalizedImage.expandDims(0);
  
    const predictions = model.predict(inputTensor);
    const topPrediction = predictions.as1D().argMax().dataSync()[0];
  
    return topPrediction;
  }
  
  function loadImage(event) {
    const imageUpload = event.target;
    const previewImage = document.getElementById("previewImage");
    const classifyButton = document.getElementById("classifyButton");
  
    if (imageUpload.files && imageUpload.files[0]) {
      const reader = new FileReader();
  
      reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewImage.style.display = "block";
        classifyButton.disabled = false;
      };
  
      reader.readAsDataURL(imageUpload.files[0]);
    }
  }
  
  async function onClassifyButtonClick() {
    const previewImage = document.getElementById("previewImage");
    const predictionResult = document.getElementById("predictionResult");
  
    console.log("Clasificando imagen...");
  
    const topPrediction = await classifyImage(previewImage);
    const classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'];
  
    predictionResult.innerText = `Predicción: ${classNames[topPrediction]}`;
  
    console.log("Predicción:", classNames[topPrediction]);
  }
  
  document.getElementById("imageUpload").addEventListener("change", loadImage);
  document.getElementById("classifyButton").addEventListener("click", onClassifyButtonClick);
  