const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
    video.play();
    startPredictionLoop();
})
.catch(err => {
    alert("Camera access required");
    console.error(err);
});

// Continuous prediction
function startPredictionLoop() {
    setInterval(() => {

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = canvas.toDataURL("image/jpeg");

        fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: image })
        })
        .then(res => res.json())
        .then(data => {
            document.getElementById("result").innerText =
                "Prediction: " + data.prediction;
        });

    }, 500); // every 0.5 sec
}