opt = {
  nThread = 2,
  logPath = 'log/toy', -- output path for log files
  dataSize = 'small',
  epMax = 30,  -- max epoches
  teFreq = 2, -- test every teFreq epoches
  batSize = 256, -- batch size
  isCuda = false,
  gpuInd = 1, -- gpu #
  C = 50,   -- #channels
  V = 30000, -- #vocabulary
  fnData = 'data/toy.lua', -- filie name for data generator
  fnModel = 'net/cmd.lua', -- file name for model
  stOptim =  {
    learningRate = 0.01,
    weightDecay = 0.0005,
    momentum = 0.9,
  },
  shrinkFreq = 100, -- shrink every # iteration
}

dofile('main.lua')