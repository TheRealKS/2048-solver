var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var Move;
(function (Move) {
    Move[Move["UP"] = 0] = "UP";
    Move[Move["DOWN"] = 1] = "DOWN";
    Move[Move["LEFT"] = 2] = "LEFT";
    Move[Move["RIGHT"] = 3] = "RIGHT";
})(Move || (Move = {}));
var gameMap = [];
var currentTimestep = -1;
var numtimesteps = 0;
function readFile(file) {
    return new Promise(function (resolve, reject) {
        var fr = new FileReader();
        fr.onload = function (x) { return resolve(fr.result); };
        fr.readAsText(file);
    });
}
function read(input) {
    return __awaiter(this, void 0, void 0, function () {
        var log, strLog, lines, state, buildingArray, arr, _i, lines_1, l, m, numbers, lnumbers, lnumbers;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, readFile(input.files[0])];
                case 1:
                    log = _a.sent();
                    document.getElementById("episodes").innerHTML = "Reading file...";
                    strLog = log;
                    lines = strLog.split("\n");
                    state = null;
                    buildingArray = 0;
                    arr = [];
                    for (_i = 0, lines_1 = lines; _i < lines_1.length; _i++) {
                        l = lines_1[_i].trim();
                        if (l.startsWith("Move")) {
                            m = l.split(".")[1];
                            state = {
                                move: Move[m],
                                state: undefined
                            };
                        }
                        else {
                            numbers = l.substr(2);
                            if (buildingArray < 3) {
                                numbers = numbers.substring(0, numbers.length - 1);
                                lnumbers = numbers.split(" ");
                                arr.push(lnumbers.filter(function (r) { return r != ""; }).map(function (r) { return parseInt(r); }));
                                buildingArray++;
                            }
                            else if (buildingArray == 3) {
                                numbers = numbers.substring(0, numbers.length - 2);
                                lnumbers = numbers.split(" ");
                                arr.push(lnumbers.filter(function (r) { return r != ""; }).map(function (r) { return parseInt(r); }));
                                state.state = arr;
                                gameMap.push(state);
                                arr = [];
                                buildingArray = 0;
                            }
                        }
                    }
                    console.log(gameMap);
                    numtimesteps = gameMap.length;
                    initUI();
                    return [2 /*return*/];
            }
        });
    });
}
function initUI() {
    var container = document.getElementById("episodes");
    container.innerHTML = "";
    gameMap.forEach(function (val, i) {
        container.appendChild(buildTimeStepUIElement(i, val.move));
    });
    selectTimestep(0, true, true);
}
function buildGridUIElement(grid) {
    var gridel = document.createElement("table");
    gridel.className = "game_grid";
    gridel.id = "game_grid";
    for (var _i = 0, grid_1 = grid; _i < grid_1.length; _i++) {
        var row = grid_1[_i];
        var rowel = document.createElement("tr");
        for (var _a = 0, row_1 = row; _a < row_1.length; _a++) {
            var tile = row_1[_a];
            var tileel = document.createElement("td");
            tileel.className = "game_tile";
            tileel.innerHTML = tile.toString();
            if (tile < 8 && tile > 0) {
                tileel.classList.add("tile_low");
            }
            else if (tile < 128 && tile > 0) {
                tileel.classList.add("tile_medlow");
            }
            else if (tile < 1024 && tile > 0) {
                tileel.classList.add("tile_medhigh");
            }
            else if (tile > 0) {
                tileel.classList.add("tile_high");
            }
            rowel.appendChild(tileel);
        }
        gridel.appendChild(rowel);
    }
    return gridel;
}
function buildTimeStepUIElement(index, move, corr) {
    if (corr === void 0) { corr = true; }
    var d = document.createElement("div");
    d.className = "timestep";
    d.id = "timestep_" + index.toString();
    d.innerHTML += index.toString() + ": ";
    var dot = document.createElement("div");
    dot.classList.add("dot");
    if (corr) {
        dot.classList.add("green");
    }
    else {
        dot.classList.add("red");
    }
    d.appendChild(dot);
    d.innerHTML += Move[move];
    d.addEventListener("click", function () {
        selectTimestep(index, true);
    });
    return d;
}
function selectTimestep(index, update, first) {
    if (update === void 0) { update = false; }
    if (first === void 0) { first = false; }
    if (index >= 0 && index < numtimesteps) {
        var newcurrent = document.getElementById("timestep_" + index.toString());
        newcurrent.classList.add("timestep_active");
        if (!first) {
            document.getElementById("timestep_" + currentTimestep.toString()).classList.remove("timestep_active");
        }
        document.getElementById("game_grid").remove();
        var gridcontainer = document.getElementById("grid_container");
        gridcontainer.insertBefore(buildGridUIElement(gameMap[index].state), gridcontainer.firstChild);
        newcurrent.scrollIntoView({ block: "center" });
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
window.onload = function () {
    document.getElementById("stepForward").addEventListener("click", function () {
        stepForward();
    });
    document.getElementById("stepBackward").addEventListener("click", function () {
        stepBackward();
    });
};
