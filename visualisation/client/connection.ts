enum Move {
    UP,
    DOWN,
    LEFT,
    RIGHT
}

interface Coordinate {
    x : number,
    y : number
}

interface GameState {
    move : Move,
    replaced : boolean,
    tileAdded : Coordinate,
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
            let ab = l.split(";")
            let m = ab[0].split(".")[1]
            state = {
                move: Move[m],
                replaced: (ab[1] === "True"),
                state: undefined,
                tileAdded: undefined
            };
        } else if (l.length == 5 || l.length == 7) {
            let coords = l.substr(1);
            coords = coords.substring(0, coords.length - 1);
            let loosecoords = coords.split(" ");
            state.tileAdded = {x: parseInt(loosecoords[1]), y: parseInt(loosecoords[0])}
        } else { 
            let numbers = l.substr(1);
            if (buildingArray == 0) {
                numbers = numbers.substr(1);
            }
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
        container.appendChild(buildTimeStepUIElement(i, val.move, val.move == Move.DOWN || val.move == Move.LEFT, val.replaced));
    });

    selectTimestep(0, true, true);
}

function buildGridUIElement(grid : Array<Array<Number>>, newtile : Coordinate) {
    let gridel = document.createElement("table");
    gridel.className = "game_grid";
    gridel.id ="game_grid";

    var r = 0
    for (var row of grid) {
        let rowel = document.createElement("tr");
        var t = 0
        for (var tile of row) {
            let tileel = document.createElement("td");
            tileel.className = "game_tile";
            if (newtile.x == t && newtile.y == r) tileel.classList.add("outline")
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
            t += 1
        }
        gridel.appendChild(rowel);
        r += 1
    }

    return gridel;
}

function buildTimeStepUIElement(index : number, move : number, corr = true, replaced = false) {
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
    d.innerHTML += replaced ? "; REPLACED" : "";

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
        gridcontainer.insertBefore(buildGridUIElement(gameMap[index].state, gameMap[index].tileAdded), gridcontainer.firstChild);

        newcurrent.parentElement.scroll(0, newcurrent.offsetTop - 55);
        if (index == numtimesteps - 1) {
            document.getElementById("game_grid").style.border = "2px solid red";
        }
    }
    if (update) {
        currentTimestep = index;
    }
}

function stepForward() {
    if (currentTimestep >= 0 && currentTimestep < numtimesteps - 1) {
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
    document.getElementById("reset").addEventListener("click", () => {
        location.reload();
    });
    document.getElementById("override").addEventListener("change", () => {
        if (document.getElementById("override").checked) {
            let inputs = document.getElementById("manual_inputs")
            for (var input of inputs.children) {
                input.disabled = false;
            }
        } else {
            let inputs = document.getElementById("manual_inputs")
            for (var input of inputs.children) {
                input.disabled = true;
            }
        }
    });

    document.getElementById("left").addEventListener("click", () => {
        manual(Move.LEFT);
    });
    document.getElementById("right").addEventListener("click", () => {
        manual(Move.RIGHT);
    });
    document.getElementById("up").addEventListener("click", () => {
        manual(Move.UP);
    });
    document.getElementById("down").addEventListener("click", () => {
        manual(Move.DOWN);
    });

    document.getElementById("startVideo").addEventListener("click", () => {
        startVideo();
    });
    document.getElementById("stopVideo").addEventListener("click", () => {
        stopVideo();
    })

    document.body.addEventListener('keydown', function(event) {
        if (!document.getElementById("override").checked) {
            return;
        }
        switch (event.key) {
            case "ArrowLeft":
                manual(Move.LEFT);
                break;
            case "ArrowRight":
                manual(Move.RIGHT);
                break;
            case "ArrowUp":
                manual(Move.UP);
                break;
            case "ArrowDown":
                manual(Move.DOWN);
                break;
        }
    });
}