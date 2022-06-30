var timer;

function startVideo() {
    //Retrieve params
    var speed = 100 - document.getElementById("speedSlider").value;
    
    if (document.getElementById("startPosition").checked) {
        selectTimestep(0, true);
    }

    //Start
    timer = setInterval(function () {
        if (currentTimestep < numtimesteps - 1) {
            stepForward();
        }
        else {
            clearInterval(timer);
            timer = undefined;
        }
    }, 10 * speed);
}

function stopVideo() {
    if (timer) {
        clearInterval(timer);
    }
}