--[[ Data Provider for data batch
subsample the variable length sequential data
and generate fixed length data
]]--

local provider = torch.class('dataProvider')

function provider:__init(X, Y, M)
  assert(#X == Y:size(1))
  assert(Y:dim() == 1, 'tensor Y must be in size {n}')
  self.X = X
  self.Y = Y
  self.ind = torch.randperm(#X)
  self.M = M or 64
end

function provider:size()
  return #self.X
end

function provider:randperm_ind()
  self.ind = torch.randperm(#self.X)
end

function provider:get_batch(ibat, batSize)
--[[
  xx: {n, M} Tensor, instances
  yy: {n} Tensor, labels
]]--
  local ibeg = batSize*(ibat - 1) + 1
  local iend = math.min(#self.X, ibeg + batSize - 1)
  local sz = iend - ibeg + 1
  assert(sz >= 1)
  
  local xx = torch.ones(sz, self.M, self.X[1]:type())
  local yy = torch.zeros(sz, self.Y:type())
  
  local j = 1
  for ii = ibeg, iend do
    local i = self.ind[ii]
    
    -- xx: copy the length M string at random position
    local function copy_sub()
      -- NOTE: 0-base for raw data pointer
      local src = self.X[i]:data()
      local srcLen = self.X[i]:numel()
      local srcBeg = math.random(srcLen) - 1
      local srcEnd = math.min(srcBeg + self.M, srcLen)
      
      local dst = xx[j]:data()
      
      local iSrc, iDst = srcBeg, 0
      while iSrc < srcEnd do 
        dst[iDst] = src[iSrc]
        iSrc, iDst = iSrc + 1, iDst + 1
      end
    end
    copy_sub()
    
    -- yy: copy
    yy[j] = self.Y[i]
    --
    j = j + 1
  end
  
  return xx, yy
end

--function provider:get_datum(i)
--  local xx = self.X[i]:clone()
--  local yy = torch.FloatTensor(1):type(self.Y:type())
--  yy:fill(self.Y[i])
--  return xx, yy
--end