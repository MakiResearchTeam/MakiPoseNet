

NUMBER_OF_KEYPOINTS = 23

CONNECT_INDEXES_FOR_PAFF =  [
    # head
    [1, 2],
    [2, 4],
    [1, 3],
    [3, 5],
    # body
    # left
    [1, 7],
    [7, 9],
    [9, 11],
    [11, 22],
    [11, 23],
    # right
    [1, 6],
    [6, 8],
    [8, 10],
    [10, 20],
    [10, 21],
    # center
    [1, 0],
    [0, 12],
    [0, 13],
    # legs
    # left
    [13, 15],
    [15, 17],
    [17, 19],
    # right
    [12, 14],
    [14, 16],
    [16, 18],
    # Additional limbs
    [5, 7],
    [4, 6],
]


