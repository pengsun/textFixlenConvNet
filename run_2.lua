opt = {
  nThread = 2,
  logPath = 'log/imdb/cmx2catd_C50_M64', -- output path for log files
  dataSize = 'full',
  epMax = 500,  -- max epoches
  teFreq = 5, -- test every teFreq epoches
  batSize = 256, -- batch size
  isCuda = true,
  gpuInd = 2, -- gpu #
  C = 50,   -- #channels
  V = 30000, -- #vocabulary
  M = 64, -- a fixed length for the string
  fnData = 'data/imdb.lua', -- filie name for data generator
  fnModel = 'net/cmx2catd.lua', -- file name for model
  fnTest = 'testVote.lua',
  stOptim =  {
    learningRate = 0.1,
    weightDecay = 0.0005,
    momentum = 0.9,
  },
  shrinkFreq = 100, -- shrink every # iteration
}

dofile('main.lua')