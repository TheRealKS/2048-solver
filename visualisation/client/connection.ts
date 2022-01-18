enum Move {
    UP,
    DOWN,
    LEFT,
    RIGHT
}

interface GameState {
    move : Move,
    state : Array<Array<Number>>
}

var gameMap : Array<GameState>= [];
var currentTimestep = -1;
var numtimesteps = 0;

function readFile(file) {
    return new Promise((resolve, reject) => {
      let fr = new FileReader();
      fr.onload = x=> resolve(fr.result);
      fr.readAsText(file);
})}

async function read(input) {
    var log = await readFile(input.files[0]);
    document.getElementById("episodes").innerHTML = "Reading file...";
    var strLog : String = <String>log;
    var lines = strLog.split("\n");
    var state : GameState = null;

    var buildingArray = 0;
    var arr : Array<Array<Number>> = [];
    for (var line of lines) {
        var l = line.trim();
        if (l.startsWith("Move")) {
            let m = l.split(".")[1]
            state = {
                move: Move[m],
                state: undefined
            };
        } else {
            let numbers = l.substr(2);
            if (buildingArray < 3) {
                numbers = numbers.substring(0, numbers.length - 1);
                let lnumbers = numbers.split(" ");
                arr.push(lnumbers.filter(r => r != "").map(r => parseInt(r)))
                buildingArray++;
            } else if (buildingArray == 3) {
                numbers = numbers.substring(0, numbers.length - 2);
                let lnumbers = numbers.split(" ");
                arr.push(lnumbers.filter(r => r != "").map(r => parseInt(r)))
                state.state = arr;
                gameMap.push(state);
                arr = [];
                buildingArray = 0;
            }
        }
    }

    console.log(gameMap)
    numtimesteps = gameMap.length;
    initUI();
}

function initUI() {
    let container = document.getElementById("episodes");
    container.innerHTML = "";

    gameMap.forEach(function(val, i) {
        container.appendChild(buildTimeStepUIElement(i, val.move));
    });

    selectTimestep(0, true, true);
}

function buildGridUIElement(grid : Array<Array<Number>>) {
    let gridel = document.createElement("table");
    gridel.className = "game_grid";
    gridel.id ="game_grid";

    for (var row of grid) {
        let rowel = document.createElement("tr");
        for (var tile of row) {
            let tileel = document.createElement("td");
            tileel.className = "game_tile";
            tileel.innerHTML = tile.toString();
            if (tile < 8 && tile > 0) {
                tileel.classList.add("tile_low");
            } else if (tile < 128 && tile > 0) {
                tileel.classList.add("tile_medlow");
            } else if (tile < 1024 && tile > 0) {
                tileel.classList.add("tile_medhigh");
            } else if (tile > 0) {
                tileel.classList.add("tile_high");
            }
            rowel.appendChild(tileel);
        }
        gridel.appendChild(rowel);
    }

    return gridel;
}

function buildTimeStepUIElement(index : number, move : number, corr = true) {
    let d = document.createElement("div");
    d.className = "timestep";
    d.id = "timestep_" + index.toString();
    d.innerHTML += index.toString() + ": ";
    
    let dot = document.createElement("div");
    dot.classList.add("dot");
    if (corr) {
        dot.classList.add("green");
    } else {
        dot.classList.add("red");
    }
    d.appendChild(dot);
    d.innerHTML += Move[move];

    d.addEventListener("click", function() {
        selectTimestep(index, true)
    });
    
    return d;
}

function selectTimestep(index : number, update = false, first = false) {
    if (index >= 0 && index < numtimesteps) {
        let newcurrent = document.getElementById("timestep_" + index.toString());
        newcurrent.classList.add("timestep_active");
        if (!first) {
            document.getElementById("timestep_" + currentTimestep.toString()).classList.remove("timestep_active");
        }

        document.getElementById("game_grid").remove();
        let gridcontainer = document.getElementById("grid_container");
        gridcontainer.insertBefore(buildGridUIElement(gameMap[index].state), gridcontainer.firstChild);

        newcurrent.scrollIntoView({block: "center"})
    }
    if (update) {
        currentTimestep = index;
    }
}

function stepForward() {
    if (currentTimestep >= 0 && currentTimestep < numtimesteps) {
        selectTimestep(currentTimestep + 1);
        currentTimestep++;
    }
}

function stepBackward() {
    if (currentTimestep >= 1) {
        selectTimestep(currentTimestep - 1);
        currentTimestep--;
    }
}

window.onload = function() {
    document.getElementById("stepForward").addEventListener("click", () => {
        stepForward();
    });
    document.getElementById("stepBackward").addEventListener("click", () => {
       stepBackward(); 
    });
}