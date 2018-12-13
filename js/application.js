 // UserMedia DefaultSize: 640x480
let originalMediaWidth = 640;
let originalMediaHeight = 480;

let tracker;
let videoWidth = 350;
let videoHeight = videoWidth * (originalMediaHeight / originalMediaWidth);
let adjustVideoWidth = originalMediaWidth / videoWidth;
let adjustVideoHeight = originalMediaHeight / videoHeight;
let canvasSize = 100;

let model;
const SMILE_NUMBER = 3; // smileと定義されたCategory番号
let probability = 1.0; //初期感情データセット
let not_smile_counter = 0; //笑顔じゃない場合の連続数

// html要素挿入
document.body.innerHTML += `
 <video id="inputVideo" width="${videoWidth}" height="${videoHeight}" loop preload autoplay></video>
<canvas id="inputVideoCanvas" width="${canvasSize}" height="${canvasSize}"></canvas>
<div id="few_smile"><br /><br />笑顔がたりないよ！<br />頑張れ頑張れ！٩(•౪• ٩)</div>
`

// 学習済みモデルのロード
async function loadModel(){
  console.log('[Dehehe]Model Loading...')
  model = await tf.loadModel(chrome.extension.getURL('trained_data/output/model.json'))
  console.log('[Dehehe]Model Loaded!!')
};

// トラッカー準備
function initTracker() {
  tracker = new tracking.ObjectTracker('face');
  tracker.setInitialScale(4);
  tracker.setStepSize(2);
  tracker.setEdgesDensity(0.1);
  tracking.track('#inputVideo', tracker, { camera: true, fps: 4 });
};

// トラッキング状態監視
async function startTracking(){
  tracker.on('track', function(event) {
    if(!event.data) return;
    event.data.forEach(function(rect) {
      let input_video_canvas_data = getInputVideoCanvasData(rect);
      let input_video_canvas_tensor = convertToTensor(input_video_canvas_data);
      predict(input_video_canvas_tensor);
      if(probability > 0.88) {
        not_smile_counter = 0
        document.getElementById("few_smile").classList.add("to_hidden")
        document.getElementById("few_smile").classList.remove("to_show")
      }else if(not_smile_counter++ > 4){
        document.getElementById("few_smile").classList.add("to_show")
        document.getElementById("few_smile").classList.remove("to_hidden")
      }
    });
  });
}

// Video To Canvas
function getInputVideoCanvasData(rect){
  let inputVideo = document.getElementById('inputVideo');
  let inputVideoCanvas = document.getElementById('inputVideoCanvas');
  inputVideoCanvas.getContext('2d').drawImage(
    inputVideo, 
    rect.x * adjustVideoWidth, 
    rect.y * adjustVideoHeight, 
    rect.width * adjustVideoWidth, 
    rect.height * adjustVideoHeight,
    0,
    0,
    canvasSize,
    canvasSize
  );
  return inputVideoCanvas;
}

// canvasデータをTensor形式に変換
function convertToTensor(canvas){
  let tensor = tf.fromPixels(canvas, 1).resizeNearestNeighbor([64,64]).toFloat();
  let offset = tf.scalar(255);
  tensor_image = tensor.div(offset).expandDims();

  return tensor_image;
}

// 学習モデルによる推定
async function predict(tensor){
  let prediction = await model.predict(tensor).data();
  let result = Array.from(prediction).map(function(p,i){
    return {
      probability: p,
      classNumber: i
    };
  }).filter(obj => obj.classNumber == SMILE_NUMBER)
  .shift()

  probability = result.probability
};


loadModel(); // モデルロード呼び出し
initTracker(); // トラッカー準備
startTracking(); // トラッキング状態監視 & 推定
