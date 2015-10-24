
-- check & prepare upvalues
assert(md, 'upvalue model not set')
local opt = opt or {
  isCuda = true,
  batSize = 256,
}

local test = function (data, theInfo, ep)
  -- check input
  assert(data, 'data provider not set')
  assert(theInfo, 'training info not set')
  assert(ep, 'epoch not set')
  
  
  print('testing epoch ' .. ep)
  md:evaluate()  -- set testing (disable dropout)
  theInfo.conf:zero() -- reset/init info
  theInfo.ell[ep] = 0

  -- test each datum
  local time = sys.tic()---------------------------------
  local nb = math.ceil( data:size()/opt.batSize )
  for ibat = 1, nb do
    -- get instances-labels batch
    local inputs, targets = data:get_batch(ibat, opt.batSize)
    if opt.isCuda then 
      inputs, targets = inputs:cuda(), targets:cuda()
    end

    -- fprop
    local outputs = md:forward(inputs)
    local f = loss:forward(outputs, targets)

    -- update error, loss
    theInfo.conf:batchAdd(outputs, targets)
    theInfo.ell[ep] = theInfo.ell[ep] + f*inputs:size(1)

    -- print
    xlua.progress(ibat, nb)
    -- print debug info
    --print(input:size())
    --print_flow()
  end -- for i
  time = sys.toc(time)-----------------------------------

  -- update error, loss
  theInfo.conf:updateValids()
  theInfo.err[ep] = 1 - theInfo.conf.totalValid
  theInfo.ell[ep] = theInfo.ell[ep] / data:size()
  -- print
  print(theInfo.conf)
  print(string.format('ell = %f, err = %d %%',
      theInfo.ell[ep], theInfo.err[ep]*100))
  print(string.format('time = %ds, speed = %d data/s, or %f ms/data',
      time, data:size()/time, time/data:size()*1000))
end

return test