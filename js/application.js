// Trackingの実装とTensor変換部分については下記記事を参考
// https://github.com/PonDad/manatee/tree/master/2_emotion_recognition-master

let model;
let originalVideoWidth = 640;
let videoWidth = 320;
let videoHeight = (videoWidth / 4) * 3; //比率固定
let emotion= [3, 1.0]; //初期スマイルデータ用 3はsmile
let tracker = new tracking.LandmarksTracker();

// 今回の本筋でないので、わけないでゴリ書き
$('body').append(`
<div style="position: fixed; right: 10px; top: 50px; z-index:9999">
  <video id="video" width="${videoWidth}" height="${videoHeight}" style="position: fixed; right: -${videoWidth}px" loop preload autoplay></video>
  <canvas id="video-canvas" width="100" height="100"></canvas>
</div>
<div id="few_smile"><br /><br />笑顔がたりないよ！<br />頑張れ頑張れ！٩(•౪• ٩)</div>
<style>
.to_hidden{
  z-index: -1 !important;
  opacity: 0 !important;
  transition-property: all;
  transition-duration: 50ms;
  transition-delay: 0s;
  transition-timing-function: ease;
}
.to_show{
  z-index: 9999 !important;
  opacity: 1 !important;
  transition-property: all;
  transition-duration: 50ms;
  transition-delay: 0s;
  transition-timing-function: ease;
}
#few_smile{
  width: 100%;
  height: 100%;
  z-index: 9999;
  background: rgba(0, 0, 0, .7);
  color: #FFF;
  opacity: 0;
  display: block;
  top: 0%;
  text-align: center;
  position: fixed;
  font-size: 4rem;
}
</style>
`);


// videoComponent取得
let video = $('#video').get(0);

// 学習済みモデルをロードする
async function loadModel(){
  console.log("Model loading...");
  model = await tf.loadModel(`https://s3-ap-northeast-1.amazonaws.com/emotion-model/model.json`);
  console.log("Model loaded!!!");
};


function openWebStream() {
  tracker.setInitialScale(4);
  tracker.setStepSize(2);
  tracker.setEdgesDensity(0.1);
  tracking.track(video, tracker, { camera: true, fps: 8 });
};

// 顔状態解析
let not_smile_counter = 0
function alignment(){
  tracker.on('track', function(event) {
    if(!event.data) return;
    event.data.faces.forEach(function(rect) {
      predict(rect);
      // console.log(emotion[1])
      if(emotion[0] == 3 && emotion[1] > 0.88) {
        not_smile_counter = 0
        $('#few_smile').addClass('to_hidden').removeClass('to_show')
      }else if(not_smile_counter++ > 4){
        $('#few_smile').addClass('to_show').removeClass('to_hidden')
      }
    });
  });
};

// 推測
async function predict(rect){
  let tensor = captureWebcamera(rect) ;

  let prediction = await model.predict(tensor).data();
  let results = Array.from(prediction).map(function(p,i){
    return {
      probability: p,
      classNumber: i
    };
  }).sort(function(a,b){
    return b.probability-a.probability;
  }).slice(0,6);

  results.forEach(function(p){
    return emotion = [results[0].classNumber, results[0].probability] 
  });
};

// ウェブカメラからのデータを取得してCanvasに転写
// 転写時にサイズも学習済みモデル用にサイズ変更
function captureWebcamera(rect) {
  let faceCanvas = $('#video-canvas').get(0);
  let faceContext = faceCanvas.getContext('2d');

  let adjust = originalVideoWidth / video.width
  start_x = rect.x * adjust
  start_y = rect.y * adjust
  face_width = rect.width * adjust
  face_height = rect.width * adjust
  faceContext.drawImage(video, start_x, start_y, face_width, face_height, 0, 0, 100, 100);
  let tensor = tf.fromPixels(faceCanvas, 1).resizeNearestNeighbor([64,64]).toFloat();
  let offset = tf.scalar(255);
  tensor_image = tensor.div(offset).expandDims();

  return tensor_image;
}

// モデルロード呼び出し
loadModel();

// ウェブカメラ準備呼び出し
openWebStream();

// 顔追跡処理 呼び出し
alignment();
