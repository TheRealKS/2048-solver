var board = undefined;
function setupManual() {
    moveBoard(Move.DOWN);
}
function moveBoard(action) {
    if (!board) {
        var temp = gameMap[currentTimestep].state;
        board = nj.array(temp);
    }
    var sum_merges = 0;
    board = transform_board(board, action, true);
    for (var i = 0; i < board.shape[0]; i++) {
        var row = board.slice([i, null]);
        console.log(row);
        var non_zero = [];
        for (var _i = 0; _i < row.length; _i++) {
            var tile = row[_i];
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
        row = non_zero;
        for (var k = 0; k < (4 - non_zero.length); k++) {
            row.push(0);
        }
        board.set(i, null, row);
    }

    for (var row of board) {
        console.log(row);
    }
    return sum_merges;
}
function transform_board(board, direction, forward) {
    if (forward) {
        if (direction == Move.UP || direction == Move.DOWN)
            board = board.T;
        if (direction == Move.DOWN || direction == Move.RIGHT)
            board = board.slice(null, [null, null, -1]);
    }
    else {
        if (direction == Move.DOWN || direction == Move.RIGHT)
            board = board.slice(null, [null, null, -1]);
        if (direction == Move.UP || direction == Move.DOWN)
            board = board.T;
    }
    return board;
}
