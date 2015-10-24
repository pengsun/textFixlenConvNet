
-- check & prepare upvalues
assert(md, 'upvalue model not set')
opt = opt or {
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
  data:range_ind() -- use natural index order
  theInfo.conf:zero() -- reset/init info
  theInfo.ell[ep] = 0

  -- test each doc
  local time = torch.tic()---------------------------------
  local nb = math.ceil( data:size()/opt.batSize )
  for idoc = 1, data:size() do
    -- get instances-labels batch
    local inputs, targets = data:get_batch_doc(idoc)
    if opt.isCuda then 
      inputs, targets = inputs:cuda(), targets:cuda()
    end
    
    -- test each fixed length string and vote
--    if inputs:size(1) == 1 then
--      require('mobdebug').start()
--    end
    -- fprop
    local outputs = md:forward(inputs)
    local f = loss:forward(outputs, targets)
    -- vote
    local pred = outputs:sum(1):squeeze()
    --update loss 
    theInfo.ell[ep] = theInfo.ell[ep] + f*inputs:size(1)
    
    --[[ test one by one
    local pred
    for ii = 1, inputs:size(1) do
      -- fprop
      local output = md:forward(inputs[ii])
      -- accumulate
      if ii == 1 then 
        pred = output:clone() 
      else 
        pred:add(output)
      end
      local f = loss:forward(output, targets[ii])
      -- update loss
      theInfo.ell[ep] = theInfo.ell[ep] + f
    end
    ]]--
    
    -- update error
    theInfo.conf:add(pred, targets[1])

    -- print
    xlua.progress(idoc, data:size())
    -- print debug info
    --print(input:size())
    --print_flow()
  end -- for ibat
  if opt.isGpu then cutorch.synchronize() end
  time = torch.toc(time)-----------------------------------

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