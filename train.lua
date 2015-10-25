require'optim'
require'xlua'

-- check & prepare upvalues
assert(md, 'upvalue model not set')
local param, gradParam = md:getParameters()

assert(loss, 'upvalue loss not set')

opt = opt or {
  isCuda = true,
  batSize = 256,
  stOptim =  {
    learningRate = 0.01,
    weightDecay = 0.0005,
    momentum = 0.9,
  },
  shrinkFreq = 25, -- shrink every # iteration
}
local stOptim = opt.stOptim
local shrinkFreq = opt.shrinkFreq


local train = function (data, theInfo, ep)
  -- check input
  assert(data, 'data provider not set')
  assert(theInfo, 'training info not set')
  assert(ep, 'epoch not set')

  print('training epoch ' .. ep)
  md:training() -- set training (enable dropout)
  data:randperm_ind() -- random shuffling instances
  -- reset/init info
  theInfo.conf:zero()
  theInfo.ell[ep] = 0
  -- shrink learningRate when necessary
  if ep % shrinkFreq == 0 then 
    stOptim.learningRate = stOptim.learningRate / 2
    print('perform learningRate shrinking... New rate: ' ..
      stOptim.learningRate)
  end

  -- sgd over each batch
  local time = torch.tic()-------------------------------------
  local timeData, timeEval = 0, 0
  local nb = math.ceil( data:size()/opt.batSize )
  for ibat = 1, nb do
    -- get instances-labels batch
    local tmp1 = torch.tic() ---------------------------------
    local inputs, targets = data:get_batch(ibat, opt.batSize)
    if opt.isCuda then 
      inputs, targets = inputs:cuda(), targets:cuda()
      cutorch.synchronize()
    end
    timeData = timeData + torch.toc(tmp1) --------------------

    -- closure doing all
    local feval = function (tmp)
      gradParam:zero()
      -- fprop
      local outputs = md:forward(inputs)
      local f = loss:forward(outputs, targets)
--      print_flow()
      
      -- bprop
      local gradOutputs = loss:backward(outputs, targets)
      md:backward(inputs, gradOutputs)

      -- TODO: L1 L2 penality

      -- update error, loss
      theInfo.conf:batchAdd(outputs, targets)
      theInfo.ell[ep] = theInfo.ell[ep] + f*inputs:size(1)

      -- print debug info
--        local str = '%d: out = (%f, %f), ' .. 
--                    'f = %f, acc ell = %f'
--        print(string.format(str, 
--            i, output[1], output[2],
--            f, info.tr.ell[ep]))
      --
      return f, gradParam
    end

    -- update parameters 
    local tmp2 = torch.tic()--------------------
    optim.sgd(feval, param, stOptim)
    if opt.isGpu then cutorch.synchronize() end
    timeEval = timeEval + torch.toc(tmp2)-------
    
    -- print
    xlua.progress(ibat, nb)
    -- print debug info
    --print(input:size())
    --print_flow()
  end -- for ibat
  time = torch.toc(time)---------------------------------------

  -- update error, loss
  theInfo.conf:updateValids()
  theInfo.err[ep] = 1 - theInfo.conf.totalValid
  theInfo.ell[ep] = theInfo.ell[ep] / data:size()
  -- print
  --print(info.tr.conf)
  print(string.format('ell = %f, err = %d %%',
      theInfo.ell[ep], theInfo.err[ep]*100))
  print(string.format('time = %fs, speed = %d data/s, or %f ms/data',
      time, data:size()/time, time/data:size()*1000))
  print(string.format('time data = %fs, time eval = %fs',
      timeData, timeEval))
end -- trian

return train