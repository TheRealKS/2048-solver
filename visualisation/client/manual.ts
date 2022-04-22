var manualBoard : Array<Array<Number>>;
var timestepused = 0;

function manual(action) {
    if (!manualBoard || timestepused != currentTimestep) {
        manualBoard = gameMap[currentTimestep].state;
        timestepused = currentTimestep;
    }
    let move = moveBoard(manualBoard, action);
    manualBoard = move[0];
    let n = {x: 0, y: 0};
    if (manualBoard.every((el: Array<Number>) => el.every(t => t > 0))) {
        alert("Game over");
    } else {
        n = getRandom(manualBoard);
        manualBoard[n.y][n.x] = 2;
    }
    document.getElementById("game_grid").remove();
    let gridcontainer = document.getElementById("grid_container");
    gridcontainer.insertBefore(buildGridUIElement(manualBoard, n), gridcontainer.firstChild);
}   

function getRandom(arr) : Coordinate {
    var tileval;
    do {
        var y = Math.floor((Math.random() * (arr.length)));
        var x = Math.floor((Math.random() * (arr[y].length)));
        tileval = arr[y][x];
    } while (tileval != 0);
    return {
        x: x,
        y: y
    };
 }

function moveBoard(board, action) {
    var sum_merges = 0;
    board = transform_board(board, action, true);
    for (var i = 0; i < board.length; i++) {
        var row = board[i];
        var non_zero = [];
        for (var n = 0; n < row.length; n++) {
            var tile = row[n];
            if (tile != 0)
                non_zero.push(tile);
        }
        var j = 0;
        while (j < non_zero.length - 1) {
            if (non_zero[j] == non_zero[j + 1]) {
                non_zero[j] += non_zero[j + 1];
                sum_merges += non_zero[j];
                non_zero.splice(j + 1);
            }
            j += 1;
        }
        let i_ = 4 - non_zero.length;
        for (var k = 0; k < i_; k++) {
            non_zero.push(0);    
        }
        board[i] = non_zero;
    }

    board = transform_board(board, action, false);

    return [board, sum_merges];
}

function transform_board(board, direction, forward) {
    if (forward) {
        if (direction == Move.UP || direction == Move.DOWN)
            board = transpose(board);
        if (direction == Move.DOWN || direction == Move.RIGHT)
            board = reverseRows(board);
    }
    else {
        if (direction == Move.DOWN || direction == Move.RIGHT)
            board = reverseRows(board);
        if (direction == Move.UP || direction == Move.DOWN)
            board = transpose(board);
    }
    return board;
}

function reverseRows(arr) {
    for (var i = 0; i < arr.length; i++) {
        let r = arr[i];
        r = r.reverse();
        arr[i] = r;
    }
    return arr;
}

function transpose(arr) {
    var t = arr[0].map((col, i) => arr.map(row => row[i]));
    return t;
}