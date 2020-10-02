#include <vector>

#ifndef PAFPROCESS
#define PAFPROCESS

const float THRESH_HEAT = 0.05;
const float THRESH_VECTOR_SCORE = 0.17; // 0.05
const int THRESH_VECTOR_CNT1 = 6;
// By default THRESH_PART_CNT = 4 and THRESH_HUMAN_SCORE = 0.3,
// BUT in the original paper of the OpenPose that are below were used
// THRESH_PART_CNT = 3 and THRESH_HUMAN_SCORE = 0.2
const int THRESH_PART_CNT = 4;
const float THRESH_HUMAN_SCORE = 0.3;
const int NUM_PART = 24;

const int STEP_PAF = 10;

// Main and additional sizes of the limbs connection array
const int ADDITIONAL_SIZE = 2;
const int MAIN_SIZE = 23;
// Size of the row controller for construction of the skelet
const int SIZE_ROW = MAIN_SIZE + 2;
// Main + additional sizes
const int COCOPAIRS_SIZE = MAIN_SIZE + ADDITIONAL_SIZE; // 25

const int COCOPAIRS_NET[COCOPAIRS_SIZE][2] = {
    // Connect paff at X and Y axis into one
    // Main
    {0, 1},   {2, 3},   {4, 5},   {6, 7},   {8, 9},   {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19},
    {20, 21}, {22, 23}, {24, 25}, {26, 27}, {28, 29}, {30, 31}, {32, 33}, {34, 35}, {36, 37}, {38, 39},
    {40, 41}, {42, 43}, {44, 45},
    // Additional
    {46, 47}, {48, 49},
};

const int COCOPAIRS[COCOPAIRS_SIZE][2] = {
    // Main connection of the skelet (limbs)
    {1, 2}, {2, 4}, {1, 3}, {3, 5}, {1, 7}, {7, 9}, {9, 11}, {11, 22}, {11, 23}, {1, 6}, {6, 8}, {8, 10},
    {10, 20}, {10, 21}, {1, 0}, {0, 12}, {0, 13}, {13, 15}, {15, 17}, {17, 19}, {12, 14}, {14, 16}, {16, 18},
    // Additional connections
    {5, 7}, {4, 6},
};


struct Peak {
    int x;
    int y;
    float score;
    int id;
};

struct VectorXY {
    float x;
    float y;
};

struct ConnectionCandidate {
    int idx1;
    int idx2;
    float score;
    float etc;
};

struct Connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
};

int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3, float *pafmap);
int get_num_humans();
int get_part_cid(int human_id, int part_id);
float get_score(int human_id);
int get_part_x(int cid);
int get_part_y(int cid);
float get_part_score(int cid);

#endif
