--[[ main logic to do training & testing ]]--

require 'optim'
require 'xlua'
require 'pl.path'
require 'pl.file'

--[[ global options ]]--
opt = opt or {
  nThread = 2,
  logPath = 'log/toy', -- output path for log files
  dataSize = 'small',
  epMax = 10,  -- max epoches
  teFreq = 2, -- test every teFreq epoches
  batSize = 256, -- batch size
  isCuda = true,
  gpuInd = 1, -- gpu #
  C = 1024,   -- #channels
  V = 30000, -- #vocabulary
  M = 128, -- a fixed length for the string
  fnData = 'data/nonono.lua', -- filie name for data generator
  fnModel = 'net/nonono.lua', -- file name for model
  fnTest = 'test.lua', -- file name for testing
  stOptim =  {
    learningRate = 1,
    learningRateDecay = 1e-7,
    weightDecay = 0.0005,
    momentum = 0.5,
  },
  shrinkFreq = 4, -- shrink every # iteration
}
print('[global options]')
print(opt)
if opt.isCuda then 
  require('cunn')
  print('switch to CUDA')
  cutorch.setDevice(opt.gpuInd)
  print('use GPU #' .. opt.gpuInd)
end
print('\n')

--[[ data ]]--
local trData, teData = dofile(opt.fnData)

--[[ net ]]--
md, loss, print_flow = dofile(opt.fnModel)
if opt.isCuda then 
  md:cuda(); loss:cuda();
end

--[[ observer: log, display... ]]
info, logger, write_opt_net = dofile 'observer.lua'
write_opt_net(opt, md)

--[[ train & test ]]--
local epMax = opt.epMax or 3
param, gradParam = md:getParameters()
trainOneEpoch = dofile'train.lua'
testOneEpoch = dofile(opt.fnTest)
for ep = 1, epMax do
  trainOneEpoch(trData, info.tr, ep)
  
  if ep % opt.teFreq == 0 then -- do testing
    print('\n')
    testOneEpoch(teData, info.te, ep)
  end
  print('\n')
  
  -- move stuff in info to logger
  logger.ell:add{info.tr.ell[ep]}
  logger.err:add{info.te.err[ep]}
  
  -- plot
  logger.ell:style{'lp'}; logger.ell:plot();
  logger.err:style{'-'}; logger.err:plot();
end -- for ep