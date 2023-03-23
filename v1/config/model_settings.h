#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

// LSTM => DNN
#define kTimeSteps          20
// dim of data in each timestep
#define kNumOfDim           3
#define kInstanceSize      (kTimeSteps * kNumOfDim)

#define kCategoryCount  2
#endif // MODEL_SETTINGS_H_